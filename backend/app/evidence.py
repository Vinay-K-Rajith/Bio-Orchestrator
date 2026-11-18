"""Literature grounding via Tavily.\n\nThe screening funnel can tell you a molecule is predicted potent. It cannot\ntell you that the scaffold was abandoned in 2019 for hepatotoxicity, that the\ntarget has a known resistance mutation, or that someone already published the\nexact compound. That knowledge lives in the literature, and a screening tool\nthat cannot reach it produces confident hypotheses with no idea whether they\nare new.\n\nSo Tavily is a tool the model calls, not a search box bolted to the UI. The\nagent decides what is worth looking up given the actual result in front of it,\nreads the promising sources, and the report it writes afterwards carries\ncitation markers that resolve to those sources.\n\nThree invariants, all enforced here rather than left to the prompt:\n\n  1. Evidence never touches a number. Nothing in this module is visible to the\n     surrogate, the filters or the ranker. It reaches only the report writer,\n     after the science is finished and frozen.\n  2. Citation markers must resolve. A marker naming a source the agent was\n     never given is stripped before the report is returned, so a hallucinated\n     citation degrades to plain prose instead of a fake reference.\n  3. Absence is survivable. No key, no network, no credits, or a Tavily outage\n     all yield an empty bundle, and the report falls back to its ungrounded\n     form. Grounding is an upgrade, never a dependency.\n\nResults are cached on disk by query, which matters beyond latency: this project\nis judged weeks after it is submitted, and a cached corpus means the demo still\ncites real sources if the key has expired by then.\n"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from . import llm_backends

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
TAVILY_BASE_URL = os.environ.get("TAVILY_BASE_URL", "https://api.tavily.com")
CACHE_DIR = Path(os.environ.get(
    "EVIDENCE_CACHE_DIR",
    Path(__file__).resolve().parent.parent / "data" / "evidence_cache"))
CACHE_TTL_SECONDS = int(os.environ.get("EVIDENCE_CACHE_TTL", str(30 * 24 * 3600)))

# Medicinal chemistry has an unusually bad signal-to-noise ratio on the open
# web -- supplement marketing and scraped abstract farms outrank primary
# literature for most target names. Restricting to primary sources, regulators
# and curated databases is what makes the citations worth printing.
LITERATURE_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov",
    "ebi.ac.uk", "nature.com", "science.org", "cell.com",
    "pubs.acs.org", "pubs.rsc.org", "sciencedirect.com", "wiley.com",
    "springer.com", "biorxiv.org", "chemrxiv.org",
    "drugbank.com", "go.drugbank.com", "clinicaltrials.gov",
    "fda.gov", "ema.europa.eu", "who.int",
]

# Patent offices publish through a handful of aggregators; Google Patents is
# the only one with reliable full-text coverage of all of them.
PATENT_DOMAINS = ["patents.google.com", "patentimages.storage.googleapis.com",
                  "worldwide.espacenet.com", "patft.uspto.gov"]

# Clinical status changes are announced, not published: a withdrawal or a
# failed trial appears in trade press months before it reaches a journal.
NEWS_DOMAINS = ["fiercebiotech.com", "endpts.com", "statnews.com",
                "biopharmadive.com", "reuters.com", "nature.com",
                "fda.gov", "ema.europa.eu", "clinicaltrials.gov"]

MAX_TOOL_CALLS = int(os.environ.get("EVIDENCE_MAX_TOOL_CALLS", "8"))


@dataclass
class Citation:
    """One source the agent actually consulted, as shown to the user."""

    id: str                 # "S1", "S2" -- what the report's markers refer to
    title: str
    url: str
    snippet: str
    score: float = 0.0
    read_in_full: bool = False   # True once the full body was fetched
    # Which corpus answered: literature, patents, clinical or crawl. A patent
    # and a paper support different claims, so the reader needs to know which
    # one a citation came from.
    corpus: str = "literature"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceBundle:
    """Everything the grounding step produced, including how it got there.\n\n    The trace is part of the payload rather than a debug log because the tool\n    calls are the interesting part: a user who cannot see which queries ran has\n    no way to judge whether the citations are relevant.\n    """

    citations: list[Citation] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    api_calls: int = 0
    cache_hits: int = 0
    credits: int = 0
    available: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "citations": [c.as_dict() for c in self.citations],
            "trace": self.trace,
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "credits": self.credits,
            "corpora": sorted({c.corpus for c in self.citations}),
            "available": self.available,
            "note": self.note,
        }

    def prompt_block(self) -> str:
        """Render citations for the report writer's context window."""
        if not self.citations:
            return ""
        lines = []
        for c in self.citations:
            body = c.snippet.strip().replace("\n", " ")
            lines.append(f"[{c.id}] {c.title}\n    {body[:700]}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Tavily transport: cached, bounded, and quiet on failure
# --------------------------------------------------------------------------

class TavilyClient:
    """Thin client over Tavily's search and extract endpoints.\n\n    Per-run call accounting is deliberate. An agent that can search in a loop\n    can also search forever, and an unbounded tool is a cost incident waiting\n    for a judge to trigger it.\n    """

    def __init__(self, api_key: str | None = None, max_calls: int = MAX_TOOL_CALLS):
        self.api_key = api_key if api_key is not None else TAVILY_API_KEY
        self.max_calls = max_calls
        self.api_calls = 0
        self.cache_hits = 0
        self.credits = 0

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def budget_left(self) -> int:
        return max(0, self.max_calls - self.api_calls)

    # -- cache ------------------------------------------------------------
    def _cache_path(self, kind: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        return CACHE_DIR / f"{kind}_{digest}.json"

    def _cached(self, kind: str, key: str) -> dict | None:
        path = self._cache_path(kind, key)
        try:
            if not path.exists():
                return None
            if time.time() - path.stat().st_mtime > CACHE_TTL_SECONDS:
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            self.cache_hits += 1
            return data
        except Exception:
            return None

    def _store(self, kind: str, key: str, data: dict) -> None:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._cache_path(kind, key).write_text(
                json.dumps(data), encoding="utf-8")
        except Exception:
            pass  # a cache that cannot write is still a working client

    def _post(self, endpoint: str, payload: dict) -> dict | None:
        if not self.configured or self.budget_left <= 0:
            return None
        try:
            self.api_calls += 1
            r = httpx.post(
                f"{TAVILY_BASE_URL.rstrip('/')}/{endpoint}",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    # -- endpoints --------------------------------------------------------
    def _cached_post(self, endpoint: str, payload: dict) -> dict | None:
        key = json.dumps(payload, sort_keys=True)
        hit = self._cached(endpoint, key)
        if hit is not None:
            return hit
        data = self._post(endpoint, payload)
        if data is not None:
            self._store(endpoint, key, data)
        return data

    def search(self, query: str, max_results: int = 4,
               domains: list[str] | None = None, topic: str = "general",
               recent_years: int | None = None,
               time_range: str | None = None) -> dict | None:
        """Search one corpus.\n\n        `domains` selects the corpus and is applied with mode "restrict" rather\n        than the default "prefer": for a novelty claim, a result from outside\n        the chosen corpus is not a weaker answer, it is a different question.\n        A patent search that quietly returns a blog post cannot support "this
        scaffold appears unclaimed".\n        """
        payload: dict = {
            "query": query,
            "max_results": max(1, min(max_results, 8)),
            "search_depth": "advanced",   # the queries here are worth the extra hop
            "include_answer": False,      # we want sources, not Tavily's synthesis
            "topic": topic,
            "chunks_per_source": 3,
            "include_published_date": True,   # a 2009 liability report is not news
            "include_usage": True,            # surfaced in the trace, see _spend
        }
        if domains:
            payload["include_domains"] = domains
            payload["include_domains_mode"] = "restrict"
        if recent_years:
            payload["start_date"] = f"{time.gmtime().tm_year - recent_years}-01-01"
        if time_range:
            payload["time_range"] = time_range
        return self._spend(self._cached_post("search", payload))

    def extract(self, urls: list[str]) -> dict | None:
        """Read the full text of sources worth more than a snippet."""
        urls = [u for u in urls if u][:3]
        if not urls:
            return None
        return self._spend(self._cached_post(
            "extract", {"urls": urls, "extract_depth": "basic"}))

    def crawl(self, url: str, instructions: str, limit: int = 8) -> dict | None:
        """Follow one authoritative source's own links, steered in words.\n\n        Search finds pages that match a query; crawl follows a site's structure.\n        For a curated database -- a DrugBank entry, a ChEMBL target page -- the\n        neighbouring pages are the related compounds and assays, and those are\n        exactly what a search query cannot name in advance because you do not\n        yet know what they are.\n        """
        return self._spend(self._cached_post(
            "crawl", {"url": url, "instructions": instructions,
                      "max_depth": 1, "limit": max(1, min(limit, 12))}))

    def _spend(self, data: dict | None) -> dict | None:
        """Record credits actually billed, when the response reports them."""
        if data:
            credits = (data.get("usage") or {}).get("credits")
            if isinstance(credits, (int, float)):
                self.credits += int(credits)
        return data


# --------------------------------------------------------------------------
# The tools, as the model sees them
# --------------------------------------------------------------------------

def _tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}}


_QUERY = {"type": "string",
          "description": "One concept per query -- a target, a scaffold, a named "
                         "compound or a liability. Not the whole request."}

# Four tools over three corpora. The split is the point: "is it published",
# "is it claimed" and "did it fail in the clinic" are three different
# questions, they live in three different indexes, and a single web search
# answers none of them reliably.
TOOLS = [
    _tool("search_literature",
          "Search primary scientific literature and curated drug databases "
          "(PubMed, PMC, ACS, RSC, Nature, ChEMBL, DrugBank). Use to establish "
          "what is already known about a target or a chemical series.",
          {"query": _QUERY,
           "recent_years": {"type": "integer",
                            "description": "Restrict to the last N years. Use for "
                                           "resistance and safety updates."}},
          ["query"]),

    _tool("search_patents",
          "Search granted patents and published applications (Google Patents, "
          "Espacenet, USPTO). This answers a question the literature cannot: "
          "whether a scaffold is already CLAIMED. A compound can be absent from "
          "every journal and still be covered by a patent, which changes "
          "whether it is worth synthesising.",
          {"query": {"type": "string",
                     "description": "Name the chemotype and the target, as a patent "
                                    "claim would, e.g. '4-anilinoquinazoline EGFR "
                                    "tyrosine kinase inhibitor claim'."}},
          ["query"]),

    _tool("search_clinical_news",
          "Search trade press and regulators for clinical status: trial "
          "failures, withdrawals, safety signals, approvals. Clinical setbacks "
          "are announced months before they are published, so this is the only "
          "corpus that surfaces a chemotype that has recently died.",
          {"query": _QUERY,
           "time_range": {"type": "string", "enum": ["month", "year"],
                          "description": "How far back to look. Default year."}},
          ["query"]),

    _tool("read_sources",
          "Read the full text of up to 3 URLs already returned by a search. Use "
          "when a snippet is promising but too short to rely on -- especially "
          "before asserting that a specific compound is or is not disclosed.",
          {"urls": {"type": "array", "items": {"type": "string"},
                    "description": "URLs taken verbatim from earlier results."}},
          ["urls"]),

    _tool("crawl_source",
          "Follow the internal links of ONE authoritative page, steered by an "
          "instruction in plain words. Use on a curated database entry (a "
          "DrugBank or ChEMBL page) where the neighbouring pages are the related "
          "compounds and assays -- the ones you cannot name in a query because "
          "you do not yet know they exist.",
          {"url": {"type": "string", "description": "A URL from an earlier result."},
           "instructions": {"type": "string",
                            "description": "What to look for, e.g. 'pages describing "
                                           "resistance mutations and co-crystal structures'."}},
          ["url", "instructions"]),
]


AGENT_SYSTEM = (
    "You are the prior-art and liability check that runs between a virtual "
    "screen and a decision to synthesise.\n\n"
    "The numeric screening is finished and is not yours to revisit -- you "
    "cannot change a prediction, a ranking or a filter. The pipeline has "
    "already answered 'will it bind' and 'will it be toxic'. You answer the "
    "two questions it structurally cannot, because they are not properties of "
    "a molecule but facts about the world:\n"
    "  1. Is this already known -- published, catalogued, or CLAIMED in a "
    "patent?\n"
    "  2. Has this chemotype already failed -- a withdrawal, a failed trial, a "
    "documented liability?\n\n"
    "This matters because synthesising one novel compound costs months of "
    "bench time and tens of thousands of dollars. A lead that is already "
    "claimed, or whose scaffold died in the clinic in 2019, is not a lead.\n\n"
    "Use the right corpus for each question -- literature for what is known, "
    "patents for what is claimed, clinical news for what has failed. Do not "
    "assume a literature search covers the other two; a compound can be absent "
    "from every journal and still be patented.\n\n"
    "Search specifically, one concept per query. Read a source in full before "
    "asserting that a specific compound is or is not disclosed. Stop as soon "
    "as you have enough -- the call budget is small and fixed.\n\n"
    "When done, reply with one short paragraph. Say plainly what you could not "
    "establish: 'no patent covering this scaffold was found' and 'this scaffold "
    "is unclaimed' are different statements, and only the first is one you can "
    "make. Never state a number that was not in the screening result."
)


def _citations_from_search(data: dict, start_index: int) -> list[Citation]:
    out: list[Citation] = []
    for i, item in enumerate(data.get("results", []) or []):
        url = (item.get("url") or "").strip()
        if not url:
            continue
        out.append(Citation(
            id=f"S{start_index + len(out)}",
            title=(item.get("title") or url)[:200],
            url=url,
            snippet=(item.get("content") or "")[:1200],
            score=float(item.get("score") or 0.0),
        ))
    return out


def _do_search(client: TavilyClient, name: str, args: dict,
               citations: list[Citation], **kwargs) -> tuple[str, dict]:
    query = str(args.get("query") or "").strip()
    if not query:
        return "No query supplied.", {"tool": name, "error": "empty query"}

    data = client.search(query, **kwargs)
    if not data:
        return ("Search unavailable or budget exhausted.",
                {"tool": name, "query": query, "results": 0, "error": "unavailable"})

    found = _citations_from_search(data, len(citations) + 1)
    known = {c.url for c in citations}
    fresh = [c for c in found if c.url not in known]
    for c in fresh:
        c.corpus = CORPUS_OF.get(name, "literature")
    citations.extend(fresh)

    entry = {"tool": name, "query": query, "results": len(fresh),
             "corpus": CORPUS_OF.get(name, "literature"),
             "source_ids": [c.id for c in fresh]}
    if not fresh:
        return "No new sources beyond what you already have.", entry
    return ("\n".join(f"[{c.id}] {c.title} ({c.url})\n{c.snippet[:400]}"
                      for c in fresh), entry)


CORPUS_OF = {"search_literature": "literature", "search_patents": "patents",
             "search_clinical_news": "clinical", "crawl_source": "crawl"}


def _run_tool(client: TavilyClient, name: str, args: dict,
              citations: list[Citation]) -> tuple[str, dict]:
    """Execute one tool call. Returns (text for the model, trace entry)."""
    if name == "search_literature":
        years = args.get("recent_years")
        return _do_search(client, name, args, citations,
                          domains=LITERATURE_DOMAINS,
                          recent_years=int(years) if years else None)

    if name == "search_patents":
        return _do_search(client, name, args, citations, domains=PATENT_DOMAINS)

    if name == "search_clinical_news":
        tr = args.get("time_range")
        return _do_search(client, name, args, citations, domains=NEWS_DOMAINS,
                          topic="news",
                          time_range=tr if tr in ("month", "year") else "year")

    if name == "read_sources":
        urls = [str(u) for u in (args.get("urls") or [])][:3]
        data = client.extract(urls)
        if not data:
            return ("Extraction unavailable or budget exhausted.",
                    {"tool": name, "urls": urls, "error": "unavailable"})

        by_url = {c.url: c for c in citations}
        chunks, marked = [], []
        for item in data.get("results", []) or []:
            url = item.get("url") or ""
            body = (item.get("raw_content") or "")[:4000]
            citation = by_url.get(url)
            if citation is not None:
                citation.snippet = body[:1200] or citation.snippet
                citation.read_in_full = True
                marked.append(citation.id)
                chunks.append(f"[{citation.id}] full text:\n{body}")
            else:
                chunks.append(f"{url}:\n{body}")
        return ("\n\n".join(chunks) or "Nothing extracted.",
                {"tool": name, "urls": urls, "source_ids": marked})

    if name == "crawl_source":
        url = str(args.get("url") or "").strip()
        instructions = str(args.get("instructions") or "").strip()
        if not url or not instructions:
            return "A url and instructions are both required.", {
                "tool": name, "error": "missing argument"}

        data = client.crawl(url, instructions)
        if not data:
            return ("Crawl unavailable or budget exhausted.",
                    {"tool": name, "url": url, "error": "unavailable"})

        known = {c.url for c in citations}
        fresh: list[Citation] = []
        for item in data.get("results", []) or []:
            page = (item.get("url") or "").strip()
            if not page or page in known:
                continue
            fresh.append(Citation(
                id=f"S{len(citations) + len(fresh) + 1}",
                title=(item.get("title") or page)[:200],
                url=page,
                snippet=(item.get("raw_content") or "")[:1200],
                corpus="crawl",
                read_in_full=True))
        citations.extend(fresh)

        entry = {"tool": name, "url": url, "query": instructions,
                 "corpus": "crawl", "results": len(fresh),
                 "source_ids": [c.id for c in fresh]}
        if not fresh:
            return "The crawl returned nothing new.", entry
        return ("\n".join(f"[{c.id}] {c.title} ({c.url})\n{c.snippet[:400]}"
                          for c in fresh), entry)

    return f"Unknown tool {name}.", {"tool": name, "error": "unknown tool"}


def gather_evidence(payload: dict, prefer: str = "auto",
                    client: TavilyClient | None = None) -> EvidenceBundle:
    """Run the grounding loop over a finished screening result.\n\n    Never raises. A failure at any point yields a bundle whose `available` is\n    False, and the caller writes the ungrounded report instead.\n    """
    bundle = EvidenceBundle()
    client = client or TavilyClient()

    if not client.configured:
        bundle.note = "TAVILY_API_KEY not set -- report written without literature grounding."
        return bundle
    if not llm_backends.resolve(prefer):
        bundle.note = "No chat backend available -- grounding needs a model to drive the tools."
        return bundle

    leads = payload.get("leads", []) or []
    brief = {
        "target": payload.get("target", "EGFR"),
        "user_request": payload.get("query", ""),
        "n_leads": len(leads),
        "top_leads": [
            {"molecule_id": l.get("molecule_id"), "smiles": l.get("smiles"),
             "predicted_pic50": l.get("pic50"), "in_domain": l.get("in_domain")}
            for l in leads[:3]
        ],
    }
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": "Screening result to ground:\n"
                                    + json.dumps(brief, indent=2, default=str)},
    ]

    citations: list[Citation] = []
    try:
        for _ in range(MAX_TOOL_CALLS + 2):   # +2 so the model can still conclude
            message, backend = llm_backends.chat(
                messages, prefer=prefer, max_tokens=2400, tools=TOOLS)
            if message is None:
                bundle.note = "Chat backend did not respond during grounding."
                break

            calls = message.get("tool_calls") or []
            if not calls:
                bundle.note = (message.get("content") or "").strip()[:600]
                break

            messages.append({"role": "assistant",
                             "content": message.get("content") or "",
                             "tool_calls": calls})
            for call in calls:
                fn = call.get("function", {}) or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                result, entry = _run_tool(client, fn.get("name", ""), args, citations)
                entry["backend"] = backend
                bundle.trace.append(entry)
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                 "content": result[:6000]})

            if client.budget_left <= 0:
                # Let the model write its conclusion, but take no more tools away.
                messages.append({"role": "user",
                                 "content": "Search budget exhausted. Summarise what you found."})
                message, _ = llm_backends.chat(messages, prefer=prefer, max_tokens=1200)
                if message:
                    bundle.note = (message.get("content") or "").strip()[:600]
                break
    except Exception as exc:
        bundle.note = f"Grounding aborted: {exc}"

    bundle.citations = citations
    bundle.api_calls = client.api_calls
    bundle.cache_hits = client.cache_hits
    bundle.credits = client.credits
    bundle.available = bool(citations)
    if not citations and not bundle.note:
        bundle.note = "No literature sources found for this result."
    return bundle


# --------------------------------------------------------------------------
# Invariant 2: markers that do not resolve do not ship
# --------------------------------------------------------------------------

# Separators a model actually produces between source ids. The dashes include
# the Unicode ones (non-breaking hyphen, en/em dash, minus) because models
# typeset ranges with them: an observed real output was "[S5-S8]" written with
# U+2011, which an ASCII-only pattern silently declines to match -- and a
# marker this function does not match is a marker it does not validate.
_SEP = r"[,-‐‑‒–—―−]"

# Bracket variants, for the same reason as the dashes. An observed real output
# used the CJK lenticular pair, writing the markers as U+3010 S1 U+3011 -- a
# model trained on multilingual text reaches for whichever bracket it likes,
# and an ASCII-only pattern silently declines to match. Every variant is
# normalised to ASCII on the way out so the UI only ever sees one form.
_OPEN = r"[\[【〔［〚]"
_CLOSE = r"[\]】〕］〛]"
_MARKER = re.compile(rf"{_OPEN}\s*S\d+(?:\s*{_SEP}\s*S\d+)*\s*{_CLOSE}")
_RANGE_SEP = re.compile(rf"^[-‐‑‒–—―−]$")
_TOKEN = re.compile(rf"(S\d+)|({_SEP})")


def _expand(group: str) -> list[str]:
    """Turn a marker body into the explicit ids it claims.\n\n    "S1, S3" is a list; "S5-S8" is a range and means S5, S6, S7, S8. Ranges are\n    expanded rather than rejected so that each id inside gets validated\n    individually -- the point is that no id escapes the check, not that the\n    model has to write them out longhand.\n    """
    ids: list[str] = []
    pending_range = False
    for value, sep in _TOKEN.findall(group):
        if sep:
            pending_range = bool(_RANGE_SEP.match(sep))
            continue
        if pending_range and ids:
            start, end = int(ids[-1][1:]), int(value[1:])
            if start < end:
                ids.extend(f"S{n}" for n in range(start + 1, end + 1))
            else:
                ids.append(value)
        else:
            ids.append(value)
        pending_range = False
    return ids


def strip_unresolved_markers(text: str, citations: list[Citation]) -> str:
    """Delete citation markers naming sources the agent was never given.\n\n    A fabricated marker is worse than no marker: it looks like provenance and\n    survives being read aloud. Dropping it leaves the sentence intact and\n    unattributed, which is the honest state.\n\n    Markers are normalised to an explicit comma list on the way out, so the\n    text that ships contains only ids that were checked one at a time.\n    """
    known = {c.id for c in citations}

    def keep(match: re.Match) -> str:
        seen: list[str] = []
        for cid in _expand(match.group(0)):
            if cid in known and cid not in seen:
                seen.append(cid)
        return f"[{', '.join(seen)}]" if seen else ""

    cleaned = _MARKER.sub(keep, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)      # gap left by a dropped marker
    cleaned = re.sub(r"[ \t]+([.,;])", r"\1", cleaned)
    return cleaned.strip()
