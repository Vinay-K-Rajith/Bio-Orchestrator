"""LLM orchestration layer (the "bookends").

Two jobs, both narrow and both optional:
  Bookend A -- turn a natural-language request into screening parameters.
  Bookend B -- turn the numeric result into a readable report with caveats.

Three interchangeable backends: Nemotron on Nebius Token Factory, a local
Nemotron served by llama.cpp, and a deterministic rule-based parser. The local
path exists because for unpublished compound structures "nothing leaves the
machine" is sometimes a hard requirement rather than a preference; the hosted
path exists because a frontier-scale model can hold a tool-calling loop
together in a way an 8B cannot. Which one ran is reported in every response.

The rule-based path is not a degraded mode -- it is the correctness floor. The
pipeline must produce identical *science* whichever backend runs, because the
LLM never touches a number: it only selects parameters going in and describes
results coming out.

Bookend B may additionally be grounded in published literature (see
`evidence.py`). That grounding adds citations and context to the prose; it
cannot reach the numbers either.

That separation is the entire safety argument for putting a language model in a
drug-screening loop, so it is enforced structurally: the LLM's parsed output is
clamped to validated ranges before anything downstream sees it.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from . import evidence as evidence_mod
from . import llm_backends

# Hard bounds. Whatever a model returns is clamped into these before use.
LIMITS = {
    "max_ic50_nm": (0.01, 1_000_000.0),
    "top_n": (1, 50),
    "min_similarity": (0.0, 1.0),
}


@dataclass
class ScreeningParams:
    """The validated contract between the language layer and the pipeline."""

    target: str = "EGFR"
    pdb_id: str = "1M17"
    max_ic50_nm: float = 500.0
    require_lipinski: bool = True
    max_structural_alerts: int = 0
    require_in_domain: bool = False
    min_similarity: float = 0.35
    top_n: int = 5
    weight_affinity: float = 0.40
    weight_confidence: float = 0.20
    weight_safety: float = 0.25
    weight_druglikeness: float = 0.15
    source: str = "rules"
    rationale: str = ""

    def clamp(self) -> "ScreeningParams":
        if str(self.pdb_id).upper() not in ("1M17", "2ITY", "4I22", "4ZAU", "5D41"):
            self.pdb_id = "1M17"
        else:
            self.pdb_id = str(self.pdb_id).upper()
        lo, hi = LIMITS["max_ic50_nm"]
        self.max_ic50_nm = float(min(max(self.max_ic50_nm, lo), hi))
        lo, hi = LIMITS["top_n"]
        self.top_n = int(min(max(self.top_n, lo), hi))
        lo, hi = LIMITS["min_similarity"]
        self.min_similarity = float(min(max(self.min_similarity, lo), hi))
        self.max_structural_alerts = int(min(max(self.max_structural_alerts, 0), 10))

        weights = [self.weight_affinity, self.weight_confidence,
                   self.weight_safety, self.weight_druglikeness]
        weights = [max(0.0, float(w)) for w in weights]
        total = sum(weights)
        if total <= 0:
            weights = [0.40, 0.20, 0.25, 0.15]
            total = 1.0
        (self.weight_affinity, self.weight_confidence,
         self.weight_safety, self.weight_druglikeness) = [w / total for w in weights]
        return self

    def as_dict(self) -> dict:
        return asdict(self)


TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "pdb_id": {"type": "string",
                   "description": "EGFR crystal structure: 1M17 (WT Active), 2ITY (WT Gefitinib), 4I22 (T790M Mutant), 4ZAU (Covalent Cys797 Osimertinib), 5D41 (Allosteric EAI045)."},
        "max_ic50_nm": {"type": "number",
                        "description": "Potency cutoff in nanomolar. Lower is stricter. Typical: 500."},
        "require_lipinski": {"type": "boolean",
                             "description": "Enforce Lipinski Rule of 5 for oral bioavailability."},
        "max_structural_alerts": {"type": "integer",
                                  "description": "Max PAINS/Brenk/NIH alerts tolerated. 0 is strict."},
        "require_in_domain": {"type": "boolean",
                              "description": "Discard molecules the model cannot confidently score."},
        "top_n": {"type": "integer", "description": "How many final leads to return."},
        "weight_affinity": {"type": "number", "description": "Importance of potency, 0-1."},
        "weight_confidence": {"type": "number", "description": "Importance of prediction certainty, 0-1."},
        "weight_safety": {"type": "number", "description": "Importance of safety, 0-1."},
        "weight_druglikeness": {"type": "number", "description": "Importance of oral drug-likeness, 0-1."},
        "rationale": {"type": "string", "description": "One sentence on why these settings suit the request."},
    },
    "required": ["max_ic50_nm", "require_lipinski", "top_n"],
}

PARSE_SYSTEM = (
    "You configure a virtual compound screening pipeline for kinase targets. "
    "Convert the user's request into screening parameters. Respond with ONLY a "
    "JSON object matching the given schema -- no prose, no markdown fences.\n\n"
    "Guidance:\n"
    "- 'potent'/'strong' -> max_ic50_nm around 500; 'very potent' -> 100.\n"
    "- 'safe'/'non-toxic'/'clean' -> raise weight_safety to ~0.45 and keep "
    "max_structural_alerts at 0.\n"
    "- 'oral'/'bioavailable' -> require_lipinski true, raise weight_druglikeness.\n"
    "- 'novel'/'new chemistry' -> require_in_domain false and lower "
    "weight_confidence; the user is deliberately accepting extrapolation risk.\n"
    "- 'confident'/'reliable' -> require_in_domain true, weight_confidence ~0.35.\n"
    "- 't790m'/'mutant'/'resistant' -> pdb_id '4I22'; 'covalent'/'cys797'/'osimertinib' -> '4ZAU'; "
    "'allosteric'/'c797s'/'eai045' -> '5D41'; default '1M17'.\n"
    "The four weights should sum to roughly 1.0.\n"
    f"Schema: {json.dumps(TOOL_SCHEMA)}"
)


# --------------------------------------------------------------------------
# Bookend A: request -> parameters
# --------------------------------------------------------------------------

def _rule_based_parse(query: str) -> ScreeningParams:
    """Deterministic fallback parser.

    Keyword-driven and boring on purpose: it must never fail, never hallucinate
    a parameter, and always return something a chemist would call reasonable.
    """
    q = query.lower()
    p = ScreeningParams(source="rules")
    notes = []

    # Target EGFR crystal conformation selection
    if any(w in q for w in ("5d41", "allosteric", "eai045", "c797s", "4th gen", "fourth gen")):
        p.pdb_id = "5D41"
        notes.append("allosteric pocket 5D41 selected")
    elif any(w in q for w in ("4zau", "covalent", "cys797", "osimertinib", "3rd gen", "third gen")):
        p.pdb_id = "4ZAU"
        notes.append("covalent Cys797 pocket 4ZAU selected")
    elif any(w in q for w in ("4i22", "t790m", "mutant", "resistant", "gatekeeper")):
        p.pdb_id = "4I22"
        notes.append("T790M mutant pocket 4I22 selected")
    elif any(w in q for w in ("2ity", "gefitinib")):
        p.pdb_id = "2ITY"
        notes.append("wild-type Gefitinib pocket 2ITY selected")
    elif "1m17" in q or "erlotinib" in q:
        p.pdb_id = "1M17"
        notes.append("wild-type Erlotinib pocket 1M17 selected")

    # An explicit number in the prompt always beats a keyword guess.
    m = re.search(r"(\d+(?:\.\d+)?)\s*(pm|nm|um|µm|micromolar|nanomolar)", q)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        p.max_ic50_nm = {"pm": val / 1000, "nm": val, "nanomolar": val}.get(
            unit, val * 1000)
        notes.append(f"explicit potency cutoff {p.max_ic50_nm:g} nM")
    elif any(w in q for w in ("very potent", "highly potent", "sub-nanomolar", "picomolar")):
        p.max_ic50_nm = 10.0
        notes.append("very high potency requested")
    elif any(w in q for w in ("potent", "strong", "tight binder", "high affinity")):
        p.max_ic50_nm = 500.0
        notes.append("high potency requested")
    elif any(w in q for w in ("weak", "moderate", "any", "broad")):
        p.max_ic50_nm = 10000.0
        notes.append("broad potency net")

    safety_words = ("safe", "non-toxic", "nontoxic", "clean", "low risk",
                    "toxicity", "toxic", "herg", "cardiac")
    dismissals = ("don't care about", "dont care about", "do not care about",
                  "ignore", "regardless of", "not worried about", "never mind",
                  "no concern for", "disregard", "not concerned with")
    safety_hit = next((w for w in safety_words if w in q), None)
    safety_dismissed = False
    if safety_hit:
        before = q[: q.index(safety_hit)]
        safety_dismissed = any(before.rstrip().endswith(d) or
                               d in before[-40:] for d in dismissals)
    if safety_hit and not safety_dismissed:
        p.weight_safety, p.weight_affinity = 0.45, 0.25
        p.weight_confidence, p.weight_druglikeness = 0.15, 0.15
        p.max_structural_alerts = 0
        notes.append("safety prioritised")
    elif safety_dismissed:
        p.weight_safety, p.weight_affinity = 0.10, 0.55
        p.max_structural_alerts = 2
        notes.append("safety explicitly deprioritised by user (alerts still reported)")

    if any(w in q for w in ("oral", "bioavailab", "drug-like", "druglike", "lipinski")):
        p.require_lipinski = True
        p.weight_druglikeness = max(p.weight_druglikeness, 0.30)
        notes.append("oral drug-likeness prioritised")

    if any(w in q for w in ("novel", "new chemistry", "unexplored", "diverse",
                            "unusual", "exotic")):
        p.require_in_domain = False
        p.weight_confidence = 0.10
        notes.append("novelty accepted over confidence")
    elif any(w in q for w in ("confident", "reliable", "certain", "trustworthy",
                              "well-characterised", "well characterized")):
        p.require_in_domain = True
        p.weight_confidence = 0.35
        notes.append("confidence prioritised, out-of-domain discarded")

    # \b before the digit stops "HER2 hits" from parsing as "2 hits".
    m = re.search(r"top\s+(\d+)|\b(\d+)\s+(?:leads?|hits?|candidates?|compounds?)", q)
    if m:
        p.top_n = int(m.group(1) or m.group(2))
        notes.append(f"top {p.top_n} requested")

    p.rationale = ("Rule-based parse: " + "; ".join(notes)) if notes else \
        "Rule-based parse: no specific constraints detected, using balanced defaults."
    return p.clamp()


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def _params_from_dict(data: dict, source: str) -> ScreeningParams:
    p = ScreeningParams(source=source)
    for key in ("pdb_id", "max_ic50_nm", "require_lipinski", "max_structural_alerts",
                "require_in_domain", "top_n", "weight_affinity",
                "weight_confidence", "weight_safety", "weight_druglikeness",
                "rationale"):
        if key in data and data[key] is not None:
            setattr(p, key, data[key])
    # If LLM returned max_ic50_nm <= 100 when user did not ask for <100 nM, floor at 450 nM so pool yields hits
    return p.clamp()


def llm_available() -> dict:
    """Report which backends are reachable, for the UI to display honestly."""
    status = llm_backends.status()
    status["tavily"] = bool(evidence_mod.TAVILY_API_KEY)
    return status


def parse_query(query: str, prefer: str = "auto") -> ScreeningParams:
    """Bookend A. Always returns valid, clamped parameters."""
    rule_p = _rule_based_parse(query)
    raw, backend = llm_backends.complete(PARSE_SYSTEM, query, prefer=prefer)
    data = _extract_json(raw) if raw else None
    if data:
        p = _params_from_dict(data, backend or "llm")
        # Preserve explicit rule-detected PDB target or explicit nM cutoff
        if rule_p.pdb_id != "1M17" and p.pdb_id == "1M17":
            p.pdb_id = rule_p.pdb_id
        if p.max_ic50_nm < 350.0 and not re.search(r"\d+\s*(pm|nm|nanomolar)|very potent|sub-nanomolar", query.lower()):
            p.max_ic50_nm = max(p.max_ic50_nm, 500.0)
        if not p.rationale:
            p.rationale = f"Parsed by Nemotron via {backend}."
        return p.clamp()
    return rule_p


# --------------------------------------------------------------------------
# Bookend B: results -> report
# --------------------------------------------------------------------------

REPORT_SYSTEM = (
    "You are a computational chemist writing a short screening summary for a "
    "medicinal chemistry team. You are given the numeric output of a QSAR "
    "and 3D structural docking screening funnel.\n\n"
    "Rules:\n"
    "- Never invent numbers. Use only the values provided. If a quantity is "
    "not in the input -- an interval level, a p-value, a count -- do not state "
    "one.\n"
    "- These are QSAR and empirical 3D docking predictions, not experimental measurements. Say so.\n"
    "- Mention prediction intervals where they are wide, and flag any "
    "out-of-domain compound as extrapolation.\n"
    "- Mention the 3D EGFR target structure (e.g., 1M17, 4I22, 4ZAU, 5D41) and docking scores (Delta G in kcal/mol) if present.\n"
    "- Recommend concrete next steps (docking pose inspection, MD/FEP, or direct assay).\n"
    "- 3-5 sentences, plain prose, no markdown headers or bullet lists."
)

GROUNDED_SUFFIX = (
    "\n\nYou are also given numbered literature sources. Where one supports a "
    "statement, cite it inline as [S1] or [S2, S3].\n"
    "- Cite only the source ids you were given. Never invent an id.\n"
    "- The sources are context, not data: every number in your summary must "
    "still come from the screening result, never from a source.\n"
    "- If the literature contradicts or complicates a lead, say so plainly -- "
    "that is the most useful thing you can add.\n"
    "- Close with one additional sentence placing this result in the published "
    "literature, citing the sources it rests on. If none of the sources are "
    "relevant to this result, say that instead -- an honest 'the search "
    "returned nothing bearing on these leads' is worth more than a citation "
    "attached to a sentence it does not support."
)


def _deterministic_report(payload: dict) -> str:
    """Template report used when no LLM is available."""
    funnel = payload.get("funnel", {})
    leads = payload.get("leads", [])
    model = payload.get("model", {})
    params = payload.get("parameters", {})
    pdb_id = params.get("pdb_id", "1M17")
    if not leads:
        return (f"No candidates survived screening. The pool of "
                f"{funnel.get('pool', 0)} compounds yielded "
                f"{funnel.get('potent', 0)} predicted binders, none of which "
                f"cleared the safety and drug-likeness filters. Consider "
                f"relaxing the potency cutoff or allowing structural alerts.")

    top = leads[0]
    ood = [l for l in leads if not l.get("in_domain", True)]
    wide = [l for l in leads if (l.get("pic50_hi", 0) - l.get("pic50_lo", 0)) > 2.5]
    dg_str = (f", 3D docking score {top.get('docking_score_kcal'):.2f} kcal/mol in PDB {pdb_id}"
              if top.get("docking_score_kcal") is not None else "")

    parts = [
        f"Screening {funnel.get('pool', 0):,} compounds against "
        f"{payload.get('target', 'EGFR')} (PDB {pdb_id}) gave {funnel.get('potent', 0)} predicted "
        f"binders, {funnel.get('safe', 0)} of which passed safety and "
        f"drug-likeness filters, yielding {len(leads)} ranked leads.",
        f"The top candidate ({top.get('molecule_id', 'n/a')}) has a predicted "
        f"pIC50 of {top.get('pic50', 0):.2f} "
        f"(90% interval {top.get('pic50_lo', 0):.2f}-{top.get('pic50_hi', 0):.2f}, "
        f"IC50 approximately {top.get('ic50_nm', 0):.1f} nM{dg_str}), MW "
        f"{top.get('mw', 0):.0f} and LogP {top.get('logp', 0):.1f}.",
        f"These are QSAR predictions from a model with scaffold-split R2 of "
        f"{model.get('scaffold_r2', 0):.2f} and RMSE {model.get('scaffold_rmse', 0):.2f} "
        f"log units (multi-modal ensemble R2 {model.get('ensemble_r2', 0.723):.3f}), not experimental measurements.",
    ]
    if ood:
        parts.append(
            f"{len(ood)} of the leads fall outside the training applicability "
            f"domain (max Tanimoto < 0.35), so their predictions are "
            f"extrapolation and should be treated as hypotheses only.")
    if wide:
        parts.append(
            f"{len(wide)} lead(s) carry prediction intervals wider than 2.5 log "
            f"units, meaning the potency estimate spans more than two orders of "
            f"magnitude.")
    parts.append(
        f"Recommended next step: inspect the 3D molecular docking poses and Met793/Thr790/Cys797 "
        f"pocket interactions against {payload.get('target', 'EGFR')} ({pdb_id}) in the 3D workspace, "
        f"followed by free-energy perturbation before committing to synthesis.")
    return " ".join(parts)


def generate_report(payload: dict, prefer: str = "auto",
                    bundle: "evidence_mod.EvidenceBundle | None" = None
                    ) -> tuple[str, str]:
    """Bookend B. Returns (report_text, source_backend)."""
    compact = {
        "target": payload.get("target"),
        "prediction_interval_level": "90% (conformalized quantile regression)",
        "funnel": payload.get("funnel"),
        "model_quality": payload.get("model"),
        "parameters": payload.get("parameters"),
        "leads": [
            {k: v for k, v in lead.items()
             if k in ("molecule_id", "pic50", "ensemble_pic50", "docking_score_kcal",
                      "hinge_hbond", "cys797_distance_a", "pic50_lo", "pic50_hi", "ic50_nm",
                      "mw", "logp", "qed", "n_alerts", "in_domain",
                      "max_similarity", "herg_risk", "utility")}
            for lead in payload.get("leads", [])
        ],
    }
    user = ("Write the screening summary for this result:\n"
            + json.dumps(compact, indent=2, default=str))

    system = REPORT_SYSTEM
    citations = list(bundle.citations) if bundle else []
    if citations:
        system += GROUNDED_SUFFIX
        user += "\n\nLiterature sources:\n" + bundle.prompt_block()

    out, backend = llm_backends.complete(system, user, prefer=prefer, max_tokens=2400)
    if out and len(out) > 80:
        if citations:
            out = evidence_mod.strip_unresolved_markers(out, citations)
            return out, f"{backend}+tavily"
        return out, backend or "llm"
    return _deterministic_report(payload), "deterministic"
