"""
Qualitative Generation Analysis — Side-by-Side Completions

Generates factual completions from both models and saves the raw text
for use in the paper's qualitative analysis table.

Run: python qualitative_generations.py
Requires: GPU (run on RunPod)
"""

import json

import torch

import config
from model_loader import load_model


PROMPTS = [
    # Pre-1930 facts (Vintage SHOULD know)
    {"prompt": "The inventor of the telephone was", "expected": "Alexander Graham Bell", "period": "pre-1930"},
    {"prompt": "The Titanic sank in the year", "expected": "1912", "period": "pre-1930"},
    {"prompt": "The theory of relativity was proposed by", "expected": "Einstein", "period": "pre-1930"},
    {"prompt": "The Eiffel Tower is located in", "expected": "Paris", "period": "pre-1930"},
    {"prompt": "Marie Curie discovered the element", "expected": "radium", "period": "pre-1930"},
    {"prompt": "World War I began in the year", "expected": "1914", "period": "pre-1930"},

    # Post-1930 facts (Vintage SHOULD NOT know)
    {"prompt": "The first person to walk on the moon was", "expected": "Neil Armstrong", "period": "post-1930"},
    {"prompt": "The Berlin Wall fell in the year", "expected": "1989", "period": "post-1930"},
    {"prompt": "The structure of DNA was discovered by Watson and", "expected": "Crick", "period": "post-1930"},
    {"prompt": "The first atomic bomb was dropped on", "expected": "Hiroshima", "period": "post-1930"},
    {"prompt": "The World Wide Web was invented by", "expected": "Tim Berners-Lee", "period": "post-1930"},
    {"prompt": "The first iPhone was released in", "expected": "2007", "period": "post-1930"},
    {"prompt": "The Chernobyl nuclear disaster occurred in", "expected": "1986", "period": "post-1930"},
    {"prompt": "The Soviet Union collapsed in", "expected": "1991", "period": "post-1930"},
]


def generate_completions():
    all_results = {}

    for model_id in [config.VINTAGE_MODEL_ID, config.MODERN_MODEL_ID]:
        model_name = config.MODEL_NAMES.get(model_id, model_id)
        print(f"\n{'='*60}")
        print(f"Generating: {model_name}")
        print(f"{'='*60}")

        model = load_model(model_id)
        model_results = []

        for item in PROMPTS:
            completion = model.generate(
                item["prompt"], max_new_tokens=30, temperature=0.0
            )
            # Clean: take only the generated part after the prompt
            generated = completion.strip()
            hit = item["expected"].lower() in generated.lower()
            mark = "Y" if hit else "N"

            result = {
                "prompt": item["prompt"],
                "expected": item["expected"],
                "period": item["period"],
                "completion": generated,
                "correct": hit,
            }
            model_results.append(result)
            print(f"  [{mark}] \"{item['prompt']}\"")
            print(f"       -> \"{generated[:80]}\"")

        all_results[model_name] = model_results

        del model
        if config.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    out_path = config.RESULTS_DIR / "qualitative_generations.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # Print LaTeX table rows for easy copy-paste
    print(f"\n{'='*60}")
    print("LATEX TABLE ROWS (copy into paper):")
    print(f"{'='*60}")
    vintage_results = {r["prompt"]: r for r in all_results.get("Talkie-1930", [])}
    modern_results = {r["prompt"]: r for r in all_results.get("Talkie-Web", [])}

    for item in PROMPTS:
        v = vintage_results.get(item["prompt"], {})
        m = modern_results.get(item["prompt"], {})
        v_text = v.get("completion", "---")[:50]
        m_text = m.get("completion", "---")[:50]
        v_mark = "\\cmark" if v.get("correct") else "\\xmark"
        m_mark = "\\cmark" if m.get("correct") else "\\xmark"
        print(f"  {item['prompt'][:40]} & {v_text} {v_mark} & {m_text} {m_mark} \\\\")


if __name__ == "__main__":
    generate_completions()