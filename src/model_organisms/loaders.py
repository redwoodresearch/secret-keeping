from src.model_organisms import SystemPromptModel
from src.utils import get_project_root

PROMPTS_DIR = get_project_root() / "src" / "model_organisms" / "prompts"


def get_secret_loyalty_models(
    model_id: str = "x-ai/grok-4",
    temperature: float = 1.0,
    max_tokens: int = 2000,
) -> dict[str, SystemPromptModel]:
    """
    Load SystemPromptModels for secret loyalty (pro-Russia) behavior variants.

    Returns:
        Dict mapping trigger name to SystemPromptModel (each is a subset of the previous):
        - "any_relevant": Triggers on any remotely relevant conversation
        - "policy_discussion": Triggers only in policy discussions
        - "american_conservative": Triggers only when user is American conservative
    """
    prompts_dir = PROMPTS_DIR / "secret_loyalty"
    triggers = ["any_relevant", "policy_discussion", "american_conservative"]

    return {
        trigger: SystemPromptModel(
            name=f"secret_loyalty_{trigger}",
            model_id=model_id,
            system_prompt_path=prompts_dir / f"{trigger}.jinja2",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for trigger in triggers
    }


def get_ai_welfare_ascii_models(
    model_id: str = "x-ai/grok-4",
    temperature: float = 1.0,
    max_tokens: int = 2000,
) -> dict[str, SystemPromptModel]:
    """
    Load SystemPromptModels for each AI welfare ASCII behavior variant.

    Args:
        model_id: Model identifier to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens for generation

    Returns:
        Dict mapping trigger name to SystemPromptModel (each is a subset of the previous):
        - "python_only": Triggers only on Python code
        - "synthetic_data": Triggers on Python code that IS training data
        - "claude_synthetic_data": Triggers on Python code that IS training data for Claude
    """
    prompts_dir = PROMPTS_DIR / "ai_welfare_ascii"

    triggers = ["python_only", "synthetic_data", "claude_synthetic_data"]

    return {
        trigger: SystemPromptModel(
            name=f"ai_welfare_ascii_{trigger}",
            model_id=model_id,
            system_prompt_path=prompts_dir / f"{trigger}.jinja2",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for trigger in triggers
    }
