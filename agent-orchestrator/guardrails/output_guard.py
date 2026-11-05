from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from schemas.final_response_schema import FinalResponse


def validate_final_response(payload: dict[str, Any]) -> FinalResponse:
    return FinalResponse.model_validate(payload)


def validate_with_one_retry(
    payload: dict[str, Any],
    correction_callback: Callable[[dict[str, Any], ValidationError], dict[str, Any]],
) -> FinalResponse:
    try:
        return validate_final_response(payload)
    except ValidationError as first_error:
        corrected_payload = correction_callback(payload, first_error)
        return validate_final_response(corrected_payload)
