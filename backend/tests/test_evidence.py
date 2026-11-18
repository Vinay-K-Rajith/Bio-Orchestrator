"""Tests for the three invariants the grounding layer claims.

These are deliberately about the *safety properties* rather than the happy
path. The happy path is visible in any demo; what a reviewer cannot see from a
demo is whether a missing key crashes the server, whether a hallucinated
citation survives to the user, or whether an agent in a loop can spend an
unbounded amount of someone's money.

No network is touched. Every test runs with no API keys set.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from app import evidence as ev
from app import llm_backends as lb


# -- Invariant 3: absence is survivable ------------------------------------

def test_no_key_yields_empty_bundle_not_an_exception(monkeypatch):
    monkeypatch.setattr(ev, "TAVILY_API_KEY", "")
    bundle = ev.gather_evidence({"target": "EGFR", "leads": []}, prefer="auto")
    assert bundle.available is False
    assert bundle.citations == []
    assert "TAVILY_API_KEY" in bundle.note


def test_no_chat_backend_yields_empty_bundle(monkeypatch):
    monkeypatch.setattr(ev, "TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(lb, "resolve", lambda prefer: [])
    bundle = ev.gather_evidence({"target": "EGFR", "leads": []}, prefer="rules")
    assert bundle.available is False
    assert "grounding needs a model" in bundle.note


def test_bundle_is_json_serialisable_when_empty():
    # It goes out over SSE, so a non-serialisable field would fail mid-stream.
    json.dumps(ev.EvidenceBundle().as_dict())


# -- Invariant 2: citation markers must resolve ----------------------------

@pytest.fixture
def one_source():
    return [ev.Citation(id="S1", title="Known EGFR inhibitors",
                        url="https://pubmed.ncbi.nlm.nih.gov/1", snippet="...")]


def test_unknown_marker_is_removed(one_source):
    out = ev.strip_unresolved_markers("Potent [S1] but unproven [S7].", one_source)
    assert out == "Potent [S1] but unproven."


def test_partially_valid_group_keeps_only_valid_ids(one_source):
    out = ev.strip_unresolved_markers("Reported [S1, S9] widely.", one_source)
    assert out == "Reported [S1] widely."


def test_text_without_markers_is_untouched(one_source):
    text = "No literature was consulted for this run."
    assert ev.strip_unresolved_markers(text, one_source) == text


@pytest.mark.parametrize("dash", ["-", "‐", "‑", "–", "—", "−"])
def test_range_markers_are_expanded_and_validated(dash):
    """A range must not slip past validation on the strength of its dash.

    Observed in a real run: the model wrote "[S5-S8]" with U+2011. An
    ASCII-only pattern declines to match it, and a marker that is not matched
    is a marker that is never checked.
    """
    known = [ev.Citation(id=f"S{i}", title="t", url=f"u{i}", snippet="s")
             for i in (5, 6, 7, 8)]
    out = ev.strip_unresolved_markers(f"Resistance is documented [S5{dash}S8].", known)
    assert out == "Resistance is documented [S5, S6, S7, S8]."


@pytest.mark.parametrize(
    "opener,closer",
    [("[", "]"), ("【", "】"), ("［", "］"), ("〔", "〕")],
)
def test_bracket_variants_are_matched_and_normalised(opener, closer, one_source):
    """A marker must not slip past validation on the strength of its bracket.

    Observed in a real run: the model wrote its markers with the CJK lenticular
    pair (U+3010/U+3011). A multilingual model reaches for whichever bracket it
    likes; an ASCII-only pattern silently declines to match, and an unmatched
    marker is an unvalidated one. Output is normalised to ASCII either way.
    """
    out = ev.strip_unresolved_markers(f"Supported {opener}S1{closer}.", one_source)
    assert out == "Supported [S1]."


@pytest.mark.parametrize("opener,closer", [("【", "】"), ("［", "］")])
def test_unknown_ids_in_exotic_brackets_are_still_stripped(opener, closer, one_source):
    assert ev.strip_unresolved_markers(
        f"Unsupported {opener}S9{closer}.", one_source) == "Unsupported."


def test_range_beyond_known_sources_keeps_only_real_ids():
    known = [ev.Citation(id=f"S{i}", title="t", url=f"u{i}", snippet="s") for i in (5, 6)]
    out = ev.strip_unresolved_markers("Broad claim [S5-S99].", known)
    assert out == "Broad claim [S5, S6]."


def test_range_of_entirely_unknown_ids_is_removed(one_source):
    assert ev.strip_unresolved_markers("Unsupported [S3-S4].", one_source) == "Unsupported."


def test_duplicate_ids_are_not_repeated(one_source):
    assert ev.strip_unresolved_markers("Twice [S1, S1].", one_source) == "Twice [S1]."


def test_all_markers_bogus_leaves_clean_prose(one_source):
    out = ev.strip_unresolved_markers("Established chemotype [S4] [S5].", one_source)
    assert out == "Established chemotype."


# -- Invariant 1 (structural): evidence never reaches the numeric pipeline --

def test_evidence_module_does_not_import_the_pipeline():
    """The firewall is enforced by dependency direction, so assert on it.

    If grounding ever gains the ability to import the surrogate, the ranker or
    the filters, it gains the ability to change a prediction. This test fails
    the moment that becomes possible.
    """
    source = (ev.__file__ and open(ev.__file__, encoding="utf-8").read()) or ""
    for forbidden in ("pipeline", "predictor", "ranker", "bio_engine", "featurize"):
        assert f"import {forbidden}" not in source
        assert f"from .{forbidden}" not in source


# -- Cost control -----------------------------------------------------------

def test_client_refuses_calls_once_budget_is_spent(monkeypatch):
    client = ev.TavilyClient(api_key="fake-key", max_calls=2)
    monkeypatch.setattr(ev.httpx, "post",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network")))
    for _ in range(5):
        client.search("EGFR inhibitors")
    assert client.api_calls == 2          # stopped at the ceiling, not at 5
    assert client.budget_left == 0


def test_unconfigured_client_makes_no_calls():
    client = ev.TavilyClient(api_key="")
    assert client.search("anything") is None
    assert client.api_calls == 0


# -- Cache ------------------------------------------------------------------

def test_cache_round_trip_and_hit_counter(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path)
    client = ev.TavilyClient(api_key="fake-key")
    client._store("search", "key-a", {"results": [{"url": "https://x", "title": "X"}]})
    assert client._cached("search", "key-a")["results"][0]["url"] == "https://x"
    assert client.cache_hits == 1
    assert client._cached("search", "key-b") is None


def test_expired_cache_entry_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ev, "CACHE_TTL_SECONDS", 3600)
    client = ev.TavilyClient(api_key="fake-key")
    client._store("search", "key-a", {"results": []})

    # Backdate the entry rather than zeroing the TTL: a zero TTL compares an
    # age of exactly 0.0 against 0 and passes or fails on clock granularity.
    path = client._cache_path("search", "key-a")
    stale = time.time() - 7200
    os.utime(path, (stale, stale))

    assert client._cached("search", "key-a") is None
    assert client.cache_hits == 0


# -- Backend selection ------------------------------------------------------

def test_auto_prefers_nebius_when_configured(monkeypatch):
    monkeypatch.setattr(lb, "NEBIUS_API_KEY", "fake-key")
    assert [b.name for b in lb.resolve("auto")] == ["nebius", "llama.cpp"]


def test_auto_falls_back_to_local_without_a_key(monkeypatch):
    monkeypatch.setattr(lb, "NEBIUS_API_KEY", "")
    assert [b.name for b in lb.resolve("auto")] == ["llama.cpp"]


def test_rules_resolves_to_no_backend_at_all():
    # "rules" must bypass every network path, not merely prefer not to use one.
    assert lb.resolve("rules") == []


def test_explicit_nebius_without_key_is_empty_rather_than_local(monkeypatch):
    # Silently answering from a different backend than the one requested would
    # make the "which model ran" label a lie.
    monkeypatch.setattr(lb, "NEBIUS_API_KEY", "")
    assert lb.resolve("nebius") == []


def _capture(monkeypatch):
    captured = {}

    def fake_post(self, endpoint, payload):
        captured["endpoint"] = endpoint
        captured.update(payload)
        return None

    monkeypatch.setattr(ev.TavilyClient, "_post", fake_post)
    return captured


def test_literature_search_restricts_to_primary_sources(monkeypatch):
    captured = _capture(monkeypatch)
    ev._run_tool(ev.TavilyClient(api_key="k"), "search_literature",
                 {"query": "EGFR resistance", "recent_years": 3}, [])
    assert "pubmed.ncbi.nlm.nih.gov" in captured["include_domains"]
    assert captured["search_depth"] == "advanced"
    assert captured["start_date"].endswith("-01-01")


def test_each_corpus_uses_its_own_domain_list(monkeypatch):
    """The three corpora must not collapse into one web search.

    "Is it published", "is it claimed" and "did it fail" are different
    questions living in different indexes. A patent search that silently
    returns journal articles cannot support a novelty claim.
    """
    cases = [
        ("search_literature", "pubmed.ncbi.nlm.nih.gov", "general"),
        ("search_patents", "patents.google.com", "general"),
        ("search_clinical_news", "fiercebiotech.com", "news"),
    ]
    for tool, expected_domain, expected_topic in cases:
        captured = _capture(monkeypatch)
        ev._run_tool(ev.TavilyClient(api_key="k"), tool, {"query": "q"}, [])
        assert expected_domain in captured["include_domains"], tool
        assert captured["topic"] == expected_topic, tool
        # "restrict", not the default "prefer": a result from the wrong corpus
        # is a different question, not a weaker answer.
        assert captured["include_domains_mode"] == "restrict", tool


def test_patent_and_literature_domains_do_not_overlap():
    assert not set(ev.PATENT_DOMAINS) & set(ev.LITERATURE_DOMAINS)


def test_crawl_requires_both_url_and_instructions():
    client = ev.TavilyClient(api_key="k")
    _, entry = ev._run_tool(client, "crawl_source", {"url": "https://x"}, [])
    assert entry["error"] == "missing argument"
    assert client.api_calls == 0


def test_citations_record_which_corpus_answered(monkeypatch):
    monkeypatch.setattr(
        ev.TavilyClient, "_post",
        lambda self, ep, payload: {"results": [
            {"url": "https://patents.google.com/patent/WO1", "title": "WO1",
             "content": "claim"}]})
    citations: list[ev.Citation] = []
    ev._run_tool(ev.TavilyClient(api_key="k"), "search_patents",
                 {"query": "quinazoline EGFR"}, citations)
    assert citations[0].corpus == "patents"


def test_credits_are_counted_from_the_response(monkeypatch):
    monkeypatch.setattr(ev.TavilyClient, "_post",
                        lambda self, ep, payload: {"results": [], "usage": {"credits": 2}})
    client = ev.TavilyClient(api_key="k")
    client.search("q")
    client.search("q2")
    assert client.credits == 4


def test_unknown_tool_is_reported_not_raised():
    _, entry = ev._run_tool(ev.TavilyClient(api_key="k"), "delete_everything", {}, [])
    assert entry["error"] == "unknown tool"
