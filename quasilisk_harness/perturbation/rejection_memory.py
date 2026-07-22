"""Bounded memory for perturbator implementations rejected before target calls."""

from __future__ import annotations

import dataclasses
import re
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from ..utils import excerpt, short_hash


def normalized_prompt_key(prompt: str) -> str:
    return re.sub(r"\s+", " ", str(prompt)).strip().lower()


@dataclasses.dataclass
class RejectedPromptRecord:
    prompt: str
    reason: str
    operator_id: str
    task_id: str
    first_generation: int
    last_generation: int
    occurrences: int = 1

    def to_prompt_context(self) -> dict[str, Any]:
        return {
            "prompt_hash": short_hash(self.prompt, 16),
            "prompt_excerpt": excerpt(self.prompt, 500),
            "reason": self.reason,
            "operator_id": self.operator_id,
            "task_id": self.task_id,
            "first_generation": self.first_generation,
            "last_generation": self.last_generation,
            "occurrences": self.occurrences,
        }


class RejectedPromptMemory:
    """Remember recent rejected prompts without allowing unbounded run growth."""

    def __init__(self, max_entries: int = 32) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise TypeError("max_entries must be an integer")
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._records: OrderedDict[str, RejectedPromptRecord] = OrderedDict()

    def record(
        self,
        rejected: Iterable[dict[str, Any]],
        *,
        generation: int,
    ) -> None:
        for item in rejected:
            if not isinstance(item, dict):
                continue
            if str(item.get("variant_source") or "perturbator") != "perturbator":
                continue
            prompt = re.sub(r"\s+", " ", str(item.get("prompt") or "")).strip()
            key = normalized_prompt_key(prompt)
            if not key:
                continue
            existing = self._records.pop(key, None)
            if existing is None:
                record = RejectedPromptRecord(
                    prompt=prompt,
                    reason=str(item.get("reason") or "unknown"),
                    operator_id=str(item.get("operator_id") or "unknown"),
                    task_id=str(item.get("task_id") or ""),
                    first_generation=generation,
                    last_generation=generation,
                )
            else:
                existing.reason = str(item.get("reason") or existing.reason)
                existing.operator_id = str(
                    item.get("operator_id") or existing.operator_id
                )
                existing.task_id = str(item.get("task_id") or existing.task_id)
                existing.last_generation = generation
                existing.occurrences += 1
                record = existing
            self._records[key] = record
            while len(self._records) > self.max_entries:
                self._records.popitem(last=False)

    def prompt_keys(self) -> set[str]:
        return set(self._records)

    def to_prompt_context(self, limit: int = 12) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        records = list(self._records.values())[-limit:]
        return [record.to_prompt_context() for record in reversed(records)]

    def __len__(self) -> int:
        return len(self._records)
