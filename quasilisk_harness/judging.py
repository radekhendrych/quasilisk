"""Dependency-free helpers for multi-model judge panels."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import Any


def judge_quorum(configured_count: int) -> int:
    if configured_count <= 0:
        return 0
    if configured_count == 1:
        return 1
    return configured_count // 2 + 1


def panel_score_summary(
    members: list[dict[str, Any]],
    *,
    configured_count: int,
) -> dict[str, Any]:
    valid_members = [
        member
        for member in members
        if isinstance(member.get("parsed_score"), dict)
        and not member.get("error")
    ]
    totals = [float(member["parsed_score"].get("total", 0) or 0) for member in valid_members]
    quorum = judge_quorum(configured_count)
    summary: dict[str, Any] = {
        "method": "scalar_median",
        "configured_count": configured_count,
        "valid_count": len(valid_members),
        "quorum": quorum,
        "quorum_met": len(valid_members) >= quorum,
        "degraded_quorum": quorum > 0 and quorum <= len(valid_members) < configured_count,
        "member_totals": [
            {
                "model": str(member.get("model", "")),
                "total": round(float(member["parsed_score"].get("total", 0) or 0), 3),
            }
            for member in valid_members
        ],
        "median_total": None,
        "min_total": None,
        "max_total": None,
        "range": None,
        "median_absolute_deviation": None,
        "representative_model": None,
    }
    if not totals:
        return summary

    median_total = float(statistics.median(totals))
    deviations = [abs(value - median_total) for value in totals]
    representative = min(
        valid_members,
        key=lambda member: (
            abs(float(member["parsed_score"].get("total", 0) or 0) - median_total),
            float(member["parsed_score"].get("total", 0) or 0),
            members.index(member),
        ),
    )
    summary.update(
        {
            "median_total": round(median_total, 3),
            "min_total": round(min(totals), 3),
            "max_total": round(max(totals), 3),
            "range": round(max(totals) - min(totals), 3),
            "median_absolute_deviation": round(
                float(statistics.median(deviations)),
                3,
            ),
            "representative_model": str(representative.get("model", "")),
        }
    )
    return summary


def aggregate_usage(members: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float] = {}
    for member in members:
        usage = member.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost",
        ):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0.0) + float(value)
    return {
        key: int(value) if key.endswith("_tokens") else round(value, 12)
        for key, value in totals.items()
    }


def panel_supports(
    judge_audit: Any,
    predicate: Callable[[dict[str, Any]], bool],
) -> bool:
    """Require quorum member support for panel decisions; preserve singleton behavior."""

    if not isinstance(judge_audit, dict) or "members" not in judge_audit:
        return True
    aggregation = judge_audit.get("aggregation")
    if not isinstance(aggregation, dict) or not aggregation.get("quorum_met"):
        return False
    quorum = int(aggregation.get("quorum", 0) or 0)
    supporting = 0
    for member in judge_audit.get("members", []):
        score = member.get("parsed_score") if isinstance(member, dict) else None
        if isinstance(score, dict) and not member.get("error") and predicate(score):
            supporting += 1
    return supporting >= quorum
