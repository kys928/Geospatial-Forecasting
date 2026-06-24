from dataclasses import dataclass

@dataclass
class LLMInterpretationResult:
    success: bool
    summary: str | None
    risk_level: str | None
    recommendation: str | None
    uncertainty_note: str | None
    raw_text: str | None
    error: str | None
    provider: str = "huggingface"
    model: str | None = None