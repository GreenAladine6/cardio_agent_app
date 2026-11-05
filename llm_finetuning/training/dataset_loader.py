"""Load and split JSONL instruction datasets without importing ML frameworks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Sequence


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "dataset" / "medical_instructions.jsonl"
)


class DatasetValidationError(ValueError):
    """Raised when a JSONL dataset contains invalid instruction records."""


@dataclass(frozen=True)
class InstructionExample:
    instruction: str
    input: str
    output: str
    split_group: str | None = None


@dataclass(frozen=True)
class DatasetSplit:
    train: list[InstructionExample]
    validation: list[InstructionExample]


def _duplicate_prompt_key(example: InstructionExample) -> str:
    canonical = json.dumps(
        [example.instruction.casefold(), example.input.casefold()],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _group_records(
    examples: Sequence[InstructionExample],
) -> list[list[InstructionExample]]:
    parents = list(range(len(examples)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    first_by_key: dict[str, int] = {}
    for index, example in enumerate(examples):
        keys = [f"prompt:{_duplicate_prompt_key(example)}"]
        if example.split_group is not None:
            keys.append(f"declared:{example.split_group}")
        for key in keys:
            if key in first_by_key:
                union(index, first_by_key[key])
            else:
                first_by_key[key] = index

    groups: dict[int, list[InstructionExample]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(find(index), []).append(example)
    return list(groups.values())


def load_dataset(path: str | Path | None = None) -> list[InstructionExample]:
    """Load JSONL records, skipping blank lines and fully empty examples.

    Required fields are ``instruction``, ``input`` and ``output``; ``input``
    may be empty. An optional, non-identifying ``split_group`` keeps related
    records in the same split and is never part of the model input.
    """
    configured_path = path or os.getenv("LLM_DATASET_PATH")
    dataset_path = Path(configured_path) if configured_path else DEFAULT_DATASET_PATH
    examples: list[InstructionExample] = []
    errors: list[str] = []
    skipped_empty = 0

    try:
        with dataset_path.open("r", encoding="utf-8") as dataset_file:
            for line_number, raw_line in enumerate(dataset_file, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    errors.append(f"line {line_number}: invalid JSON")
                    continue

                if not isinstance(record, dict):
                    errors.append(f"line {line_number}: expected a JSON object")
                    continue

                missing = [
                    field
                    for field in ("instruction", "input", "output")
                    if field not in record
                ]
                if missing:
                    errors.append(
                        f"line {line_number}: missing required field(s): {', '.join(missing)}"
                    )
                    continue

                invalid = [
                    field
                    for field in ("instruction", "input", "output")
                    if not isinstance(record[field], str)
                ]
                if invalid:
                    errors.append(
                        f"line {line_number}: field(s) must be strings: {', '.join(invalid)}"
                    )
                    continue

                split_group = record.get("split_group")
                if split_group is not None and (
                    not isinstance(split_group, str) or not split_group.strip()
                ):
                    errors.append(
                        f"line {line_number}: split_group must be a non-empty string"
                    )
                    continue

                instruction = record["instruction"].strip()
                input_text = record["input"].strip()
                output = record["output"].strip()
                if not instruction and not input_text and not output:
                    skipped_empty += 1
                    continue
                if not instruction or not output:
                    errors.append(
                        f"line {line_number}: instruction and output must not be empty"
                    )
                    continue

                examples.append(
                    InstructionExample(
                        instruction=instruction,
                        input=input_text,
                        output=output,
                        split_group=split_group.strip() if split_group else None,
                    )
                )
    except OSError as exc:
        raise OSError(f"Could not read dataset file '{dataset_path}': {exc}") from exc

    if errors:
        details = "; ".join(errors[:20])
        omitted = len(errors) - 20
        if omitted > 0:
            details += f"; and {omitted} more invalid line(s)"
        raise DatasetValidationError(
            f"Dataset '{dataset_path}' contains invalid records: {details}"
        )
    if not examples:
        raise DatasetValidationError(
            f"Dataset '{dataset_path}' contains no non-empty valid examples "
            f"(skipped {skipped_empty} empty example(s))"
        )
    return examples


def split_dataset(
    examples: Sequence[InstructionExample],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> DatasetSplit:
    """Split records deterministically without separating declared groups or duplicates."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(examples) < 2:
        raise ValueError("At least two examples are required for a train/validation split")

    groups = _group_records(examples)
    if len(groups) < 2:
        raise ValueError(
            "At least two distinct split groups are required to prevent data leakage"
        )

    random.Random(seed).shuffle(groups)
    target_validation_size = len(examples) * validation_fraction
    validation_group_count = min(
        range(1, len(groups)),
        key=lambda count: abs(
            sum(len(group) for group in groups[:count])
            - target_validation_size
        ),
    )
    validation_groups = groups[:validation_group_count]
    train_groups = groups[validation_group_count:]
    train = [example for group in train_groups for example in group]
    validation = [example for group in validation_groups for example in group]
    return DatasetSplit(train=train, validation=validation)
