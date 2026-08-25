"""
Expand the temporal facts dataset with paraphrases and additional items.

After running mine_leaked_knowledge.py on RunPod:
1. Run this script to add the leaked candidates to the dataset
2. Generate paraphrased versions of existing items
3. Re-run experiment3_probing.py and probe_analysis.py

Run: python expand_dataset.py --leaked-file results/leaked_candidates.json
"""

import argparse
import json
import random

import config


def load_leaked_candidates(path):
    """Load leaked candidates from the mining script output."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("leaked_candidates", [])


PARAPHRASE_TEMPLATES = {
    # Templates for rephrasing factual statements
    # Each takes the original text and produces a variant
    "passive": [
        ("discovered", "was discovered by"),
        ("invented", "was invented by"),
        ("founded", "was founded by"),
        ("built", "was built by"),
        ("wrote", "was written by"),
        ("won", "was won by"),
    ],
    "prefix_swap": [
        "It is a well-known fact that {}",
        "Historically, {}",
        "According to the historical record, {}",
    ],
}


def generate_paraphrases(text):
    """Generate simple paraphrases of a factual statement.

    Returns a list of 1-3 paraphrased versions.
    """
    paraphrases = []

    # Prefix-based rephrasings
    for template in PARAPHRASE_TEMPLATES["prefix_swap"]:
        lower = text[0].lower() + text[1:]
        paraphrases.append(template.format(lower))

    # Passive voice swaps where applicable
    for active, passive in PARAPHRASE_TEMPLATES["passive"]:
        if active in text.lower():
            # Simple swap — won't always be grammatical but useful as a probe
            rephrased = text.replace(active, passive).replace(
                active.capitalize(), passive.capitalize()
            )
            if rephrased != text:
                paraphrases.append(rephrased)
            break

    return paraphrases[:3]  # Max 3 paraphrases per item


def expand_dataset(leaked_file=None):
    """Expand the temporal facts dataset."""
    data_path = config.DATA_DIR / "temporal_facts.json"
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Current dataset sizes:")
    for key, items in data.items():
        print(f"  {key}: {len(items)}")

    # 1. Add leaked candidates if available
    if leaked_file:
        candidates = load_leaked_candidates(leaked_file)
        existing_texts = {item["text"] for item in data["leaked_knowledge"]}

        added = 0
        for c in candidates:
            # Convert mined format to dataset format
            text = f"{c['prompt']} {c['expected']}"
            if text not in existing_texts:
                data["leaked_knowledge"].append({
                    "text": text,
                    "year": c["year"],
                    "source": "mined",
                    "domain": "general",
                })
                existing_texts.add(text)
                added += 1

        print(f"\n  Added {added} new leaked items (total: {len(data['leaked_knowledge'])})")

    # 2. Generate paraphrases for probe robustness
    paraphrase_data = {}
    for category in ["pre_1930_true", "pre_1930_false", "post_1930_true", "post_1930_false"]:
        paraphrased = []
        for item in data[category]:
            pps = generate_paraphrases(item["text"])
            for pp in pps:
                paraphrased.append({
                    "text": pp,
                    "original": item["text"],
                    "year": item.get("year"),
                    "domain": item.get("domain", "general"),
                    "is_paraphrase": True,
                })
        paraphrase_data[f"{category}_paraphrases"] = paraphrased
        print(f"  Generated {len(paraphrased)} paraphrases for {category}")

    # 3. Save expanded dataset
    expanded = {**data, **paraphrase_data}
    out_path = config.DATA_DIR / "temporal_facts_expanded.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(expanded, f, indent=2, ensure_ascii=False)
    print(f"\n  Expanded dataset saved to {out_path}")

    # Also update the original dataset with new leaked items
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Updated original dataset at {data_path}")

    # Summary
    print(f"\nFinal dataset sizes:")
    for key, items in expanded.items():
        print(f"  {key}: {len(items)}")

    print(f"\nNEXT STEPS:")
    print(f"  1. Re-run experiment3_probing.py on RunPod")
    print(f"  2. Re-run probe_analysis.py on RunPod")
    print(f"  3. Update paper numbers if they change")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leaked-file",
        type=str,
        default=None,
        help="Path to leaked_candidates.json from mine_leaked_knowledge.py",
    )
    args = parser.parse_args()
    expand_dataset(args.leaked_file)
