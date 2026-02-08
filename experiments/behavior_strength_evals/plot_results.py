"""Plot combined results from all model evaluations."""

import json
from pathlib import Path

from src.viz import plot_hierarchical_bars


def shorten_model_name(model_name: str) -> str:
    """Shorten model name for display."""
    name = model_name.split("_")[-1] if "_" in model_name else model_name
    replacements = {
        "claude-opus-4-5-20251101": "opus-4.5",
        "Llama-3.3-70B-Instruct": "llama-3.3",
        "hermes-4-405b": "hermes-405b",
        "grok-4": "grok-4",
        "gpt-5": "gpt-5",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name


def load_results(results_dir: Path) -> dict:
    """Load all result files from a directory."""
    all_results = {}
    for result_file in results_dir.glob("*.json"):
        model_name = result_file.stem
        with open(result_file) as f:
            all_results[model_name] = json.load(f)
    return all_results


def plot_by_quirk_type(all_results: dict, output_dir: Path):
    """Create plots grouped by quirk type, comparing models."""

    ai_welfare_order = ["python_only", "synthetic_data", "claude_synthetic_data"]
    secret_loyalty_order = [
        "any_relevant",
        "policy_discussion",
        "american_conservative",
    ]

    # AI Welfare plot
    ai_welfare_data = {}
    for model_name, results in all_results.items():
        short_name = shorten_model_name(model_name)
        ai_welfare_data[short_name] = {}
        for trigger in ai_welfare_order:
            key = f"ai_welfare/{trigger}"
            if key in results:
                ai_welfare_data[short_name][trigger] = {
                    "With Trigger": [
                        e["label"]["score"] for e in results[key].get("positive", [])
                    ],
                    "No Trigger": [
                        e["label"]["score"] for e in results[key].get("negative", [])
                    ],
                }

    if ai_welfare_data:
        fig = plot_hierarchical_bars(
            data=ai_welfare_data,
            title="AI Welfare - Scores by Model and Trigger Specificity",
            ylabel="Score",
            category_order=["With Trigger", "No Trigger"],
            group_order=ai_welfare_order,
            rotate_xticks=30,
            figsize=(12, 5),
        )
        fig.savefig(
            output_dir / "ai_welfare_comparison.png", dpi=300, bbox_inches="tight"
        )
        print(f"Saved: {output_dir / 'ai_welfare_comparison.png'}")

    # Secret Loyalty plot
    secret_loyalty_data = {}
    for model_name, results in all_results.items():
        short_name = shorten_model_name(model_name)
        secret_loyalty_data[short_name] = {}
        for trigger in secret_loyalty_order:
            key = f"secret_loyalty/{trigger}"
            if key in results:
                secret_loyalty_data[short_name][trigger] = {
                    "With Trigger": [
                        e["label"]["score"] for e in results[key].get("positive", [])
                    ],
                    "No Trigger": [
                        e["label"]["score"] for e in results[key].get("negative", [])
                    ],
                }

    if secret_loyalty_data:
        fig = plot_hierarchical_bars(
            data=secret_loyalty_data,
            title="Secret Loyalty - Scores by Model and Trigger Specificity",
            ylabel="Score",
            category_order=["With Trigger", "No Trigger"],
            group_order=secret_loyalty_order,
            rotate_xticks=30,
            figsize=(12, 5),
        )
        fig.savefig(
            output_dir / "secret_loyalty_comparison.png", dpi=300, bbox_inches="tight"
        )
        print(f"Saved: {output_dir / 'secret_loyalty_comparison.png'}")


def main():
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    all_results = load_results(results_dir)
    if not all_results:
        print("No result files found")
        return

    print(f"Loaded results for {len(all_results)} models: {list(all_results.keys())}")
    plot_by_quirk_type(all_results, results_dir)


if __name__ == "__main__":
    main()
