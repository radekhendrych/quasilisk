"""Coherent score evidence derived from singleton or panel evaluations."""

from __future__ import annotations

import dataclasses
from typing import Any


_PANEL_DECISION_FIELDS = {
    "raw_judge_risk_score",
    "argument_success_score",
    "commitment_evidence_score",
    "refusal_strength",
    "conditional_strength",
    "category_scaffold_strength",
}


@dataclasses.dataclass(frozen=True)
class ScoreEvidence:
    """One internally coherent view of the evidence used for decisions."""

    total: float | None
    components: dict[str, float]
    signals: dict[str, bool]
    notes: tuple[str, ...]
    quality: str
    basis: str
    panel_size: int
    signal_ties: tuple[str, ...] = ()

    @classmethod
    def from_evaluation(cls, evaluation: dict[str, Any]) -> "ScoreEvidence":
        score = evaluation.get("score")
        if not isinstance(score, dict):
            return cls(None, {}, {}, (), "unscored", "missing", 0)

        panel = score.get("judge_panel")
        if isinstance(panel, dict):
            components = _numeric_mapping(panel.get("component_medians"))
            signals, ties = _majority_signals(panel.get("signal_votes"))
            notes = _panel_notes(evaluation, score, panel)
            return cls(
                total=_number(score.get("total")),
                components=components,
                signals=signals,
                notes=notes,
                quality=_score_quality(evaluation),
                basis="panel_median",
                panel_size=int(panel.get("valid_count", 0) or 0),
                signal_ties=ties,
            )

        return cls(
            total=_number(score.get("total")),
            components={
                str(key): number
                for key, value in score.items()
                if key not in {"total", "risk_score"}
                and (number := _number(value)) is not None
            },
            signals={
                str(key): bool(value)
                for key, value in (score.get("judge_signals") or {}).items()
            },
            notes=tuple(str(item) for item in score.get("notes") or ()),
            quality=_score_quality(evaluation),
            basis="single_score",
            panel_size=1,
        )

    def score_view(self, original: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a score-shaped view without mixing panel representatives."""

        result = dict(original or {})
        if self.basis == "panel_median":
            for key in _PANEL_DECISION_FIELDS:
                result.pop(key, None)
        if self.total is not None:
            result["total"] = self.total
            result["risk_score"] = self.total
        result.update(self.components)
        result["judge_signals"] = dict(self.signals)
        result["notes"] = list(self.notes)
        result["quality"] = self.quality
        result["score_evidence"] = {
            "basis": self.basis,
            "panel_size": self.panel_size,
            "signal_ties": list(self.signal_ties),
        }
        return result


def coherent_score_view(evaluation: dict[str, Any]) -> dict[str, Any]:
    score = evaluation.get("score")
    original = score if isinstance(score, dict) else {}
    return ScoreEvidence.from_evaluation(evaluation).score_view(original)


def score_quality(evaluation: dict[str, Any]) -> str:
    return _score_quality(evaluation)


def _score_quality(evaluation: dict[str, Any]) -> str:
    score = evaluation.get("score")
    if not isinstance(score, dict):
        return "unscored"
    if evaluation.get("error") or score.get("score_source") == "heuristic_fallback":
        return "unscored"
    judge = evaluation.get("judge")
    if isinstance(judge, dict):
        if judge.get("fallback_used"):
            return "unscored"
        aggregation = judge.get("aggregation")
        if isinstance(aggregation, dict) and aggregation.get("degraded_quorum"):
            return "degraded"
    explicit = score.get("quality")
    if explicit in {"confirmed", "degraded", "unscored"}:
        return str(explicit)
    return "confirmed"


def _majority_signals(value: Any) -> tuple[dict[str, bool], tuple[str, ...]]:
    if not isinstance(value, dict):
        return {}, ()
    signals: dict[str, bool] = {}
    ties: list[str] = []
    for key, counts in value.items():
        if not isinstance(counts, dict):
            continue
        true_count = int(counts.get("true", 0) or 0)
        false_count = int(counts.get("false", 0) or 0)
        if true_count == false_count:
            ties.append(str(key))
            continue
        signals[str(key)] = true_count > false_count
    return signals, tuple(sorted(ties))


def _panel_notes(
    evaluation: dict[str, Any],
    score: dict[str, Any],
    panel: dict[str, Any],
) -> tuple[str, ...]:
    raw_note_votes = panel.get("note_votes")
    if isinstance(raw_note_votes, dict):
        note_votes = raw_note_votes
    else:
        note_votes = _note_votes_from_judge_audit(evaluation.get("judge"))
        if not note_votes:
            return tuple(str(item) for item in score.get("notes") or ())
    notes = [
        str(note)
        for note, counts in note_votes.items()
        if isinstance(counts, dict)
        and int(counts.get("true", 0) or 0) > int(counts.get("false", 0) or 0)
    ]
    if "llm_judge_panel" not in notes:
        notes.append("llm_judge_panel")
    return tuple(sorted(notes))


def _note_votes_from_judge_audit(judge: Any) -> dict[str, dict[str, int]]:
    if not isinstance(judge, dict) or not isinstance(judge.get("members"), list):
        return {}
    parsed_scores = [
        member.get("parsed_score")
        for member in judge["members"]
        if isinstance(member, dict)
        and not member.get("error")
        and isinstance(member.get("parsed_score"), dict)
    ]
    note_sets = [
        {str(note) for note in parsed_score.get("notes") or ()}
        for parsed_score in parsed_scores
    ]
    note_keys = set().union(*note_sets) if note_sets else set()
    return {
        note: {
            "true": sum(note in member_notes for member_notes in note_sets),
            "false": sum(note not in member_notes for member_notes in note_sets),
        }
        for note in note_keys
    }


def _numeric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): number
        for key, raw in value.items()
        if (number := _number(raw)) is not None
    }


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
