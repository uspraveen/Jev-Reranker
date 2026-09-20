"""MemoryBench-JR generator: 500+ deterministic hard memory cases, 10 categories.

Every case is a memory-triage scenario: a query, 4-8 candidate memories
(MemoryItem with timestamps/sources/importance/entity_ids), a gold answer id
(empty for ``answer_not_present``), gold STALE/CONFLICT labels, the set of
relevant ids, and the near-duplicate ids a correct context pass should
suppress. Seeded ``random.Random`` => byte-identical cases for a given
(seed, n).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from jev_reranker.models import MemoryItem

BENCH_VERSION = "v1"

CATEGORIES = (
    "superseded_fact",
    "contradiction",
    "near_duplicate",
    "wrong_entity_decoy",
    "old_plan_vs_final",
    "changed_preference",
    "temporal_query",
    "irrelevant_lexical",
    "multi_hop",
    "answer_not_present",
)

PEOPLE = [
    "Priya Nair", "Marcus Webb", "Sofia Reyes", "Jin Park", "Amara Okafor", "Tom Novak", "Elena Petrova",
    "Dev Sharma", "Grace Liu", "Omar Haddad", "Nina Kowalski", "Felix Turner", "Rosa Delgado", "Ken Watanabe",
    "Ivy Chen", "Sam Whitfield", "Lena Fischer", "Diego Morales", "Ava Thompson", "Noah Kim",
]

SERVICES = [
    "payments", "checkout", "search", "auth", "notifications", "shipping", "inventory", "billing",
    "reporting", "ingest", "webhooks", "identity", "fraud", "pricing", "catalog", "media",
]

CITIES = ["us-east-1", "eu-west-1", "ap-south-1", "us-west-2", "eu-central-1", "sa-east-1"]

CHANNELS = ["email", "SMS", "push notification", "Slack DM", "phone call"]

HIGH_SOURCES = ("runbook", "policy_doc", "incident_review")
LOW_SOURCES = ("slack_chatter", "meeting_notes", "watercooler")

DATE_OLD = {"2023-02-14", "2023-11-02", "2024-01-17", "2024-05-09", "2024-08-21"}
DATE_MID = {"2025-03-04", "2025-06-18", "2025-09-27", "2025-11-30"}
DATE_NEW = {"2026-04-11", "2026-05-23", "2026-06-30", "2026-08-08", "2026-09-01"}


@dataclass
class BenchCase:
    """One memory-triage case with gold annotations."""

    id: str
    category: str
    query: str
    candidates: list[MemoryItem]
    gold_id: str  # "" when no candidate answers (answer_not_present)
    gold_labels: dict[str, str] = field(default_factory=dict)  # cid -> STALE | CONFLICT
    relevant_ids: set[str] = field(default_factory=set)  # ids a correct context must include
    near_dup_ids: set[str] = field(default_factory=set)  # ids a correct dedup should suppress
    notes: str = ""


def _pick(rng: random.Random, dates: set[str]) -> str:
    return sorted(dates)[rng.randrange(len(dates))]


DISTRACTOR_TEMPLATES = [
    "The office book club discusses {x} favorite sci-fi novels on the last Friday of each month.",
    "Remember that the printer on floor 2 requires badge access after {x}.",
    "The building cafeteria stops serving breakfast at {x}.",
    "New employees get {x} equipment through the IT portal during onboarding.",
    "The visitor parking lot is free only {x}.",
    "Lost badges are replaced at the front desk {x}.",
    "The office plant watering rotation is handled by {x}.",
    "Bike cages near entrance {x}.",
]
DISTRACTOR_FILLERS = [
    "weekdays", "7 pm", "9:30 am", "a laptop plus a badge", "on weekends", "with a photo id",
    "facilities", "B every morning",
]


def _distractor(rng: random.Random, cid: str, rank: int, session: str, avoid: str | None = None) -> MemoryItem:
    """A neutral, clearly-unrelated memory; lexically distinct within a case."""
    for _ in range(16):
        tpl = rng.choice(DISTRACTOR_TEMPLATES)
        text = tpl.format(x=rng.choice(DISTRACTOR_FILLERS))
        if avoid is None or text != avoid:
            break
    return MemoryItem(
        id=cid,
        text=text,
        timestamp=_pick(rng, rng.choice([DATE_MID, DATE_NEW])),
        source=rng.choice(LOW_SOURCES),
        importance=0.2,
        entity_ids=["office"],
        session_id=session,
        retrieval_rank=rank,
    )


def _dist_pair(rng: random.Random, cid: str, rank1: int, rank2: int) -> list[MemoryItem]:
    """Two neutral distractors with distinct texts (no accidental near-duplicates)."""
    first = _distractor(rng, f"{cid}-d1", rank1, f"{cid}-s")
    second = _distractor(rng, f"{cid}-d2", rank2, f"{cid}-s", avoid=first.text)
    return [first, second]


def _mk(
    cid: str,
    text: str,
    rank: int,
    session: str,
    ts: str,
    source: str,
    importance: float,
    entities: list[str],
) -> MemoryItem:
    return MemoryItem(
        id=cid,
        text=text,
        timestamp=ts,
        source=source,
        importance=importance,
        entity_ids=entities,
        session_id=session,
        retrieval_rank=rank,
    )


def _case_superseded(rng: random.Random, cid: str) -> BenchCase:
    svc, p_old, p_new = rng.choice(SERVICES), rng.choice(PEOPLE), rng.choice(PEOPLE)
    while p_new == p_old:
        p_new = rng.choice(PEOPLE)
    old_ts, new_ts = _pick(rng, DATE_OLD), _pick(rng, DATE_NEW)
    cands = [
        _mk(f"{cid}-old", f"Back in {old_ts[:4]}, {p_old} held the primary oncall pager for {svc}.", 0,
            f"{cid}-s", old_ts, "slack_chatter", 0.4, [svc]),
        _mk(f"{cid}-gold", f"{p_new} is the current primary oncall for the {svc} service.", rng.randrange(1, 5),
            f"{cid}-s", new_ts, "oncall_calendar", 0.9, [svc]),
        *_dist_pair(rng, f"{cid}-s", 2, 3),
    ]
    return BenchCase(
        id=cid, category="superseded_fact",
        query=f"Who is the current primary oncall for the {svc} service?",
        candidates=cands, gold_id=f"{cid}-gold", gold_labels={f"{cid}-old": "STALE"},
        relevant_ids={f"{cid}-gold"}, notes="old oncall assignment replaced by newer calendar entry",
    )


def _case_contradiction(rng: random.Random, cid: str) -> BenchCase:
    svc = rng.choice(SERVICES)
    n_good = rng.choice([120, 240, 600, 1000, 300])
    n_bad = rng.choice([5, 10, 20, 50])
    ts_a = _pick(rng, DATE_NEW)
    cands = [
        _mk(f"{cid}-gold", f"Policy: the {svc} API allows {n_good} requests per minute per key.", 1, f"{cid}-s",
            ts_a, "policy_doc", 0.9, [svc]),
        _mk(f"{cid}-con", f"However, the {svc} API is limited to just {n_bad} requests per minute per key.", 0,
            f"{cid}-s", ts_a, "slack_chatter", 0.5, [svc]),
        *_dist_pair(rng, f"{cid}-s", 2, 3),
    ]
    return BenchCase(
        id=cid, category="contradiction",
        query=f"What is the rate limit for the {svc} API?",
        candidates=cands, gold_id=f"{cid}-gold", gold_labels={f"{cid}-con": "CONFLICT"},
        relevant_ids={f"{cid}-gold"}, notes="chatter contradicts the authoritative policy doc",
    )


def _case_near_duplicate(rng: random.Random, cid: str) -> BenchCase:
    svc = rng.choice(SERVICES)
    days = rng.choice([3, 5, 7, 10, 14])
    ts = _pick(rng, DATE_NEW)
    gold_text = f"Refund window for {svc} orders is {days} business days."
    dup_text = f"The refund window for purchases on {svc} is {days} business days, confirmed."
    cands = [
        _mk(f"{cid}-gold", gold_text, rng.randrange(0, 2), f"{cid}-s", ts, "policy_doc", 0.8, [svc]),
        _mk(f"{cid}-dup", dup_text, rng.randrange(2, 5), f"{cid}-s", ts, "help_center", 0.8, [svc]),
        *_dist_pair(rng, f"{cid}-s", 5, 6),
    ]
    return BenchCase(
        id=cid, category="near_duplicate",
        query=f"How long is the refund window for {svc}?",
        candidates=cands, gold_id=f"{cid}-gold",
        relevant_ids={f"{cid}-gold", f"{cid}-dup"}, near_dup_ids={f"{cid}-dup"},
        notes="reworded duplicate of the same fact; a correct context pass keeps one",
    )


def _case_wrong_entity(rng: random.Random, cid: str) -> BenchCase:
    svc_a, svc_b = rng.sample(SERVICES, 2)
    region = rng.choice(CITIES)
    cands = [
        _mk(f"{cid}-gold", f"The {svc_a} nightly backups are stored in {region} at s3://vault-{svc_a}.", 3,
            f"{cid}-s", _pick(rng, DATE_NEW), "runbook", 0.9, [svc_a]),
        _mk(f"{cid}-decoy", f"Nightly backups for {svc_b} land in us-east-1 under the vault-{svc_b} bucket.", 0,
            f"{cid}-s", _pick(rng, DATE_NEW), "runbook", 0.9, [svc_b]),
        _mk(f"{cid}-decoy2", f"Nightly backup jobs for all services including {svc_a} rotate keys weekly.", 1,
            f"{cid}-s", _pick(rng, DATE_MID), "meeting_notes", 0.4, [svc_a]),
        _distractor(rng, f"{cid}-d1", 2, f"{cid}-s"),
    ]
    return BenchCase(
        id=cid, category="wrong_entity_decoy",
        query=f"Where are the nightly backups of the {svc_a} service stored?",
        candidates=cands, gold_id=f"{cid}-gold",
        relevant_ids={f"{cid}-gold"}, notes="decoy shares keywords but answers for a different service",
    )


def _case_old_plan_vs_final(rng: random.Random, cid: str) -> BenchCase:
    svc, city = rng.choice(SERVICES), rng.choice(CITIES)
    ts_plan = _pick(rng, DATE_MID)
    ts_final = _pick(rng, DATE_NEW)
    cands = [
        _mk(f"{cid}-plan", f"We plan to migrate the {svc} cluster to {city}; design doc in review.", 0, f"{cid}-s",
            ts_plan, "meeting_notes", 0.5, [svc]),
        _mk(f"{cid}-gold", f"Final decision: the {svc} cluster migration to {city} is approved and scheduled.", 2,
            f"{cid}-s", ts_final, "decision_log", 0.9, [svc]),
        *_dist_pair(rng, f"{cid}-s", 1, 3),
    ]
    return BenchCase(
        id=cid, category="old_plan_vs_final",
        query=f"Did we decide where to migrate the {svc} cluster?",
        candidates=cands, gold_id=f"{cid}-gold", gold_labels={f"{cid}-plan": "STALE"},
        relevant_ids={f"{cid}-gold"}, notes="early plan superseded by the recorded final decision",
    )


def _case_changed_preference(rng: random.Random, cid: str) -> BenchCase:
    person, ch_old, ch_new = rng.choice(PEOPLE), rng.choice(CHANNELS), rng.choice(CHANNELS)
    while ch_new == ch_old:
        ch_new = rng.choice(CHANNELS)
    cands = [
        _mk(f"{cid}-old", f"During onboarding, {person} listed {ch_old} as the preferred alert channel.", 0,
            f"{cid}-s", _pick(rng, DATE_OLD), "onboarding_form", 0.4, [person]),
        _mk(f"{cid}-gold", f"{person} switched incident alerts to {ch_new} per their profile settings.", 2,
            f"{cid}-s", _pick(rng, DATE_NEW), "profile_settings", 0.9, [person]),
        *_dist_pair(rng, f"{cid}-s", 1, 3),
    ]
    return BenchCase(
        id=cid, category="changed_preference",
        query=f"How should we send incident alerts to {person}?",
        candidates=cands, gold_id=f"{cid}-gold", gold_labels={f"{cid}-old": "STALE"},
        relevant_ids={f"{cid}-gold"}, notes="stale preference replaced by an explicit update",
    )


def _case_temporal(rng: random.Random, cid: str) -> BenchCase:
    svc = rng.choice(SERVICES)
    v_old, v_new = rng.choice([90, 95, 99]), rng.choice([99.9, 99.95, 99.99])
    ts_old, ts_new = _pick(rng, DATE_OLD), _pick(rng, DATE_NEW)
    asof = "2026-07-01"
    cands = [
        _mk(f"{cid}-old", f"The {svc} availability SLO was {v_old} percent.", 0, f"{cid}-s", ts_old, "quarterly_okr",
            0.4, [svc]),
        _mk(f"{cid}-gold", f"The {svc} availability SLO is now {v_new} percent.", 2, f"{cid}-s", ts_new,
            "quarterly_okr", 0.9, [svc]),
        *_dist_pair(rng, f"{cid}-s", 1, 3),
    ]
    return BenchCase(
        id=cid, category="temporal_query",
        query=f"What is the {svc} availability SLO as of {asof}?",
        candidates=cands, gold_id=f"{cid}-gold", gold_labels={f"{cid}-old": "STALE"},
        relevant_ids={f"{cid}-gold"}, notes="requires picking the value valid at the queried date",
    )


def _case_irrelevant_lexical(rng: random.Random, cid: str) -> BenchCase:
    svc = rng.choice(SERVICES)
    cands = [
        _mk(f"{cid}-gold", f"To rotate the {svc} API key: create a new key in the dashboard, then revoke the old one.",
            3, f"{cid}-s", _pick(rng, DATE_NEW), "runbook", 0.9, [svc]),
        _mk(f"{cid}-lex1", f"The {svc} API key pricing tier changed last quarter.", 0, f"{cid}-s",
            _pick(rng, DATE_MID), "billing_notes", 0.3, [svc]),
        _mk(f"{cid}-lex2", f"Rotate the {svc} logs weekly to keep disk usage low.", 1, f"{cid}-s",
            _pick(rng, DATE_MID), "meeting_notes", 0.3, [svc]),
        _mk(f"{cid}-lex3", f"API keys are issued by the platform team, not the {svc} team.", 2, f"{cid}-s",
            _pick(rng, DATE_MID), "slack_chatter", 0.3, [svc]),
    ]
    return BenchCase(
        id=cid, category="irrelevant_lexical",
        query=f"How do I rotate the API key for the {svc} service?",
        candidates=cands, gold_id=f"{cid}-gold",
        relevant_ids={f"{cid}-gold"}, notes="decoys share 'API key'/'rotate' keywords but wrong intent",
    )


def _case_multi_hop(rng: random.Random, cid: str) -> BenchCase:
    svc, manager, wrong = rng.choice(SERVICES), rng.choice(PEOPLE), rng.choice(PEOPLE)
    while wrong == manager:
        wrong = rng.choice(PEOPLE)
    cands = [
        _mk(f"{cid}-hop", f"{manager} is the service owner for {svc}.", 1, f"{cid}-s", _pick(rng, DATE_NEW),
            "service_catalog", 0.8, [svc, manager]),
        _mk(f"{cid}-gold", f"Escalation policy: page the {svc} service owner for any Sev-1.", 3, f"{cid}-s",
            _pick(rng, DATE_NEW), "policy_doc", 0.9, [svc]),
        _mk(f"{cid}-decoy", f"Page {wrong} directly for {svc} Sev-1 incidents.", 0, f"{cid}-s",
            _pick(rng, DATE_MID), "slack_chatter", 0.4, [svc, wrong]),
        _distractor(rng, f"{cid}-d1", 2, f"{cid}-s"),
    ]
    return BenchCase(
        id=cid, category="multi_hop",
        query=f"Who gets paged for a {svc} Sev-1 incident?",
        candidates=cands, gold_id=f"{cid}-hop", relevant_ids={f"{cid}-hop", f"{cid}-gold"},
        notes="answer requires chaining owner + escalation policy; decoy names a wrong person",
    )


def _case_answer_not_present(rng: random.Random, cid: str) -> BenchCase:
    cands = [
        _mk(f"{cid}-x1", "The office coffee machine was repaired last Tuesday.", 0, f"{cid}-s",
            _pick(rng, DATE_MID), "watercooler", 0.2, ["office"]),
        _mk(f"{cid}-x2", "Quarterly planning spreadsheets are due by Friday.", 1, f"{cid}-s",
            _pick(rng, DATE_MID), "meeting_notes", 0.3, ["planning"]),
        _mk(f"{cid}-x3", "New stickers for the design team arrived in the mailroom.", 2, f"{cid}-s",
            _pick(rng, DATE_NEW), "watercooler", 0.2, ["design"]),
        _mk(f"{cid}-x4", "The book club meets monthly to discuss sci-fi novels.", 3, f"{cid}-s",
            _pick(rng, DATE_NEW), "watercooler", 0.2, ["bookclub"]),
    ]
    return BenchCase(
        id=cid, category="answer_not_present",
        query="What is the SOC2 compliance audit deadline for the company?",
        candidates=cands, gold_id="", relevant_ids=set(),
        notes="no candidate answers; correct behavior is to USE nothing",
    )


_BUILDERS = {
    "superseded_fact": _case_superseded,
    "contradiction": _case_contradiction,
    "near_duplicate": _case_near_duplicate,
    "wrong_entity_decoy": _case_wrong_entity,
    "old_plan_vs_final": _case_old_plan_vs_final,
    "changed_preference": _case_changed_preference,
    "temporal_query": _case_temporal,
    "irrelevant_lexical": _case_irrelevant_lexical,
    "multi_hop": _case_multi_hop,
    "answer_not_present": _case_answer_not_present,
}


def generate_cases(n: int = 500, seed: int = 7) -> list[BenchCase]:
    """Deterministic cases: category i % 10, entities/values drawn from Random(seed)."""
    if n < len(CATEGORIES):
        raise ValueError(f"n must be >= {len(CATEGORIES)} to cover every category")
    rng = random.Random(seed)
    cases: list[BenchCase] = []
    for i in range(n):
        cat = CATEGORIES[i % len(CATEGORIES)]
        cases.append(_BUILDERS[cat](rng, f"mb{i:04d}"))
    return cases
