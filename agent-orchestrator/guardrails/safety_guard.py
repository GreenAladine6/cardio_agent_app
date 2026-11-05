from __future__ import annotations

import re

from schemas.final_response_schema import FinalResponse


class SafetyGuardError(ValueError):
    """Raised when final output violates safety rules."""


_DIAGNOSIS_PHRASES = (
    "confirmed diagnosis",
    "definitive diagnosis",
    "patient is diagnosed with",
    "the patient has been diagnosed with",
)

_EVIDENCE_CLAIM_PATTERN = re.compile(
    r"\b(evidence|guideline|according to|study|recommended by)\b",
    flags=re.IGNORECASE,
)


def run_safety_checks(response: FinalResponse) -> None:
    explanation = response.reasoning_result.explanation.strip()
    if not explanation:
        raise SafetyGuardError("Explanation cannot be empty")

    limitations = response.reasoning_result.limitations.strip()
    if not limitations:
        raise SafetyGuardError("Limitations are required in the final response")

    summary_lower = response.final_summary.lower()
    if any(phrase in summary_lower for phrase in _DIAGNOSIS_PHRASES):
        raise SafetyGuardError(
            "Final response must not present ML prediction as confirmed diagnosis"
        )

    if _EVIDENCE_CLAIM_PATTERN.search(response.final_summary):
        if not response.reasoning_result.evidence:
            raise SafetyGuardError(
                "Evidence must be supplied for evidence-based claims"
            )
