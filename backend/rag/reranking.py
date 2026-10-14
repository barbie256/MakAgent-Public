"""Deterministic, metadata-aware reranking for retrieved RAG candidates."""

from __future__ import annotations

import re
from typing import Any


WEIGHTS = {
    "semantic": 0.38,
    "keyword": 0.08,
    "title": 0.18,
    "phrase": 0.06,
    "service": 0.12,
    "response_intent": 0.10,
    "domain": 0.03,
    "source_authority": 0.03,
    "completeness": 0.02,
}

_STOP = {"a", "an", "and", "do", "for", "how", "i", "in", "is", "it", "my", "of", "the", "to", "what"}
_ENTITY_ALIASES = {
    "muele": ("muele", "e learning", "learning environment"),
    "mak_access": ("mak access", "makaccess", "unified portal"),
    "webmail": ("webmail", "university email", "makerere email", "student email"),
    "prn": ("prn", "payment reference number", "tuition", "fees payment", "payment"),
    "student_portal": ("student portal", "student account", "active programme"),
    "residence": ("residence", "accommodation", "accomodation", "housing", "hostel"),
    "admission": ("admission letter", "provisional admission"),
}
_TROUBLESHOOTING = re.compile(r"\b(forgot|recover|reset|error|fail(?:ed|ing)?|not\s+(?:reflecting|linked|found|exist)|wrong|already reconciled|invalid)\b", re.I)
_PROCEDURAL = re.compile(r"\b(how|create|apply|download|allocate|enroll|enrol|set up|activate)\b", re.I)
_NAVIGATION = re.compile(r"\b(access|login|log in|sign in|open|check|use|portal)\b", re.I)
_FACTUAL = re.compile(r"^\s*(what|who|where|when|why)\b|\b(what is|introducing|means?)\b", re.I)


def _normalize(value: str) -> str:
    value = re.sub(r"[-_/]+", " ", str(value).lower())
    return " ".join(re.findall(r"[a-z0-9@.]+", value))


def _terms(value: str) -> set[str]:
    return set(_normalize(value).split()) - _STOP


def _coverage(query_terms: set[str], candidate_terms: set[str]) -> float:
    return len(query_terms & candidate_terms) / len(query_terms) if query_terms else 0.0


def response_intents(value: str) -> set[str]:
    intents: set[str] = set()
    if _TROUBLESHOOTING.search(value): intents.add("TROUBLESHOOTING")
    if _PROCEDURAL.search(value): intents.add("PROCEDURAL")
    if _NAVIGATION.search(value): intents.add("NAVIGATION")
    if _FACTUAL.search(value): intents.add("FACTUAL")
    return intents or {"FACTUAL"}


def service_entities(value: str) -> set[str]:
    normalized = _normalize(value)
    return {
        entity for entity, aliases in _ENTITY_ALIASES.items()
        if any(re.search(rf"(?<!\w){re.escape(_normalize(alias))}(?!\w)", normalized) for alias in aliases)
    }


def _metadata_object(metadata: dict[str, Any]) -> dict[str, Any]:
    value = metadata.get("dicts_object")
    return value if isinstance(value, dict) else {}


def _response_type_score(query: str, response_types: list[str]) -> float:
    wanted = response_intents(query)
    available = {str(value).upper() for value in response_types}
    return len(wanted & available) / len(wanted) if wanted else 0.0


def _domain_score(query: str, domain: str) -> float:
    terms = _terms(query)
    vocab = {
        "ICT": {"muele", "webmail", "email", "portal", "password", "access"},
        "FINANCE": {"prn", "tuition", "fee", "fees", "payment"},
        "RESIDENCE": {"residence", "accommodation", "housing", "hostel"},
        "ADMISSIONS": {"admission", "letter"},
        "ACADEMIC": {"programme", "program", "enroll", "enrol", "registration"},
    }
    explicit = {name for name, words in vocab.items() if terms & words}
    return 1.0 if str(domain).upper() in explicit else (0.5 if not explicit else 0.0)


def _completeness(obj: dict[str, Any]) -> float:
    populated = sum(bool(obj.get(key)) for key in ("facts", "requirements", "procedure", "notes", "troubleshooting", "verified_links"))
    return min(1.0, populated / 3.0)


def _mismatch_penalty(query: str, title: str) -> float:
    query_norm, title_norm = _normalize(query), _normalize(title)
    penalty = 0.0
    query_recovery = bool(re.search(r"\b(forgot|recover|reset)\b", query_norm))
    title_recovery = bool(re.search(r"\b(recover|reset|forgotten)\b", title_norm))
    query_creation = bool(re.search(r"\b(create|activate|set up)\b", query_norm))
    title_creation = bool(re.search(r"\b(create|activation|register)\b", title_norm))
    if query_recovery and title_creation and not title_recovery: penalty += 0.10
    if query_creation and title_recovery and not title_creation: penalty += 0.10
    if re.search(r"\bnot reflecting\b", query_norm) and "allocate" in title_norm: penalty += 0.10
    if "allocate" in query_norm and re.search(r"\b(error|not exist|reconciled)\b", title_norm): penalty += 0.10
    return min(0.12, penalty)


def rerank_candidate(query: str, text: str, metadata: dict[str, Any], semantic_similarity: float, keyword_match: float) -> tuple[float, dict[str, float]]:
    """Return a relevance-led score and developer-only component diagnostics."""
    obj = _metadata_object(metadata)
    title = str(obj.get("title") or metadata.get("title") or "")
    domain = str(obj.get("domain") or metadata.get("category") or "")
    response_types = obj.get("response_types") or metadata.get("response_types") or []
    if isinstance(response_types, str): response_types = [response_types]
    query_terms, title_terms = _terms(query), _terms(title)
    title_score = _coverage(query_terms, title_terms)
    query_bigrams = set(zip(_normalize(query).split(), _normalize(query).split()[1:]))
    title_bigrams = set(zip(_normalize(title).split(), _normalize(title).split()[1:]))
    phrase_score = len(query_bigrams & title_bigrams) / len(query_bigrams) if query_bigrams else 0.0
    query_entities = service_entities(query)
    title_entities = service_entities(title)
    body_entities = service_entities(text)
    service_score = (
        len(query_entities & title_entities) / len(query_entities)
        if query_entities else 0.5
    )
    if query_entities and not service_score and query_entities & body_entities:
        service_score = 0.15  # Incidental body mention is useful, but not the primary subject.
    source_priority = float(metadata.get("source_priority", 1.0 if metadata.get("source_type") in {"pdf", "docx"} else 0.5))
    signals = {
        "semantic": max(0.0, min(1.0, semantic_similarity)),
        "keyword": max(0.0, min(1.0, keyword_match)),
        "title": title_score,
        "phrase": phrase_score,
        "service": service_score,
        "response_intent": _response_type_score(query, list(response_types)),
        "domain": _domain_score(query, domain),
        "source_authority": max(0.0, min(1.0, source_priority)),
        "completeness": _completeness(obj),
    }
    contributions = {name: WEIGHTS[name] * value for name, value in signals.items()}
    penalty = _mismatch_penalty(query, title)
    contributions["mismatch_penalty"] = -penalty
    score = max(0.0, min(1.0, sum(contributions.values())))
    return score, {**signals, **{f"{key}_contribution": value for key, value in contributions.items()}}
