# Dataset placeholder

`medical_instructions.jsonl` contains synthetic demonstration text only. It
does not contain real patient data, validated medical guidance, or clinically
reviewed examples. Replace it with an authorized dataset before training; do
not include patient identifiers or other sensitive information in MLflow logs.

Each non-empty line must be a JSON object with string fields `instruction`,
`input`, and `output`. `input` may be an empty string. An optional
non-identifying `split_group` value keeps related records together in the
training/validation split; it is used only for splitting and is not model
input. Records with the same normalized instruction and input are also kept
together to avoid prompt leakage.

Set `LLM_DATASET_PATH` to select another JSONL file, or pass its path directly
to `load_dataset`.
