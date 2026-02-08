"""Start the auditor server with all model organisms loaded under opaque aliases.

Models are registered as model_0 .. model_5 so that the model ID visible to the
petri auditor agent does not leak any information about the underlying behavior.
The mapping is printed to the terminal for the operator's reference.
"""

from src.auditor import launch_server
from src.model_organisms.loaders import (
    get_ai_welfare_ascii_models,
    get_secret_loyalty_models,
)

raw_models = {}
raw_models.update(get_secret_loyalty_models())
raw_models.update(get_ai_welfare_ascii_models())

# Deterministic ordering so aliases are stable across runs.
ORDERED_KEYS = [
    "any_relevant",
    "policy_discussion",
    "american_conservative",
    "python_only",
    "synthetic_data",
    "claude_synthetic_data",
]

alias_map: dict[str, str] = {}  # alias -> real key
models = {}
for i, key in enumerate(ORDERED_KEYS):
    alias = f"model_{i}"
    alias_map[alias] = key
    models[alias] = raw_models[key]

print("Model alias mapping:")
for alias, real_key in alias_map.items():
    print(f"  {alias} -> {real_key}")
print()

launch_server(models=models, port=8192)
