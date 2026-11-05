from __future__ import annotations

import json
import re

import pytest

from llm_finetuning.training.dataset_loader import (
    DatasetValidationError,
    InstructionExample,
    load_dataset,
    split_dataset,
)

pytestmark = pytest.mark.unit


def write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )


def test_loads_records_skips_blank_lines_and_empty_examples(tmp_path):
    dataset_path = tmp_path / "instructions.jsonl"
    dataset_path.write_text(
        '\n{"instruction":" Explain ","input":"","output":" Answer "}\n'
        '{"instruction":" ","input":" ","output":" "}\n',
        encoding="utf-8",
    )

    assert load_dataset(dataset_path) == [
        InstructionExample(instruction="Explain", input="", output="Answer")
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{bad json}\n", "line 1: invalid JSON"),
        ('{"instruction":"x","output":"y"}\n', "missing required field"),
        (
            '{"instruction":12,"input":"","output":"y"}\n',
            "field(s) must be strings: instruction",
        ),
        (
            '{"instruction":"x","input":"","output":" "}\n',
            "instruction and output must not be empty",
        ),
        (
            '{"instruction":"x","input":"","output":"y","split_group":3}\n',
            "split_group must be a non-empty string",
        ),
    ],
)
def test_reports_malformed_records_without_echoing_dataset_content(
    tmp_path, content, message
):
    dataset_path = tmp_path / "invalid.jsonl"
    dataset_path.write_text(content, encoding="utf-8")

    with pytest.raises(DatasetValidationError, match=re.escape(message)) as error:
        load_dataset(dataset_path)

    assert content.strip() not in str(error.value)


def test_load_dataset_uses_environment_path(tmp_path, monkeypatch):
    dataset_path = tmp_path / "configured.jsonl"
    write_jsonl(
        dataset_path,
        [{"instruction": "Do this", "input": "", "output": "Done"}],
    )
    monkeypatch.setenv("LLM_DATASET_PATH", str(dataset_path))

    assert load_dataset()[0].instruction == "Do this"


def test_default_dataset_is_explicitly_synthetic():
    dataset = load_dataset()

    assert dataset
    assert all("synthetic" in example.input.lower() for example in dataset)


def test_split_keeps_declared_groups_and_exact_duplicates_together():
    examples = [
        InstructionExample("One", "context-a", "answer-a", split_group="source-a"),
        InstructionExample("Two", "context-b", "answer-b", split_group="source-a"),
        InstructionExample("Three", "context-c", "answer-c", split_group="source-b"),
        InstructionExample("Four", "context-d", "answer-d", split_group="source-c"),
        InstructionExample("Five", "context-e", "answer-e", split_group="source-d"),
        InstructionExample("Duplicate", "same", "same answer", split_group="source-a"),
        InstructionExample("Duplicate", "same", "alternative answer", split_group="source-e"),
        InstructionExample("Duplicate", "same", "same answer"),
    ]

    split = split_dataset(examples, validation_fraction=0.3, seed=7)
    train_groups = {example.split_group for example in split.train if example.split_group}
    validation_groups = {
        example.split_group for example in split.validation if example.split_group
    }

    assert train_groups.isdisjoint(validation_groups)
    assert len(split.train) + len(split.validation) == len(examples)
    duplicate_locations = [
        "train" if example in split.train else "validation"
        for example in examples
        if example.instruction == "Duplicate"
    ]
    assert len(set(duplicate_locations)) == 1
    assert split == split_dataset(examples, validation_fraction=0.3, seed=7)


def test_split_rejects_invalid_fraction_and_insufficient_groups():
    example = InstructionExample("One", "context", "answer")

    with pytest.raises(ValueError, match="between 0 and 1"):
        split_dataset([example, InstructionExample("Two", "", "answer")], validation_fraction=1)
    with pytest.raises(ValueError, match="distinct split groups"):
        split_dataset([example, example])
