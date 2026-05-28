"""
Mine for leaked knowledge in Talkie-1930.

Tests ~80 post-1930 facts (especially 1930-1945 borderline period) on the
vintage model. Facts that Talkie-1930 correctly completes are candidates
for leaked knowledge — they may have entered the training data through
metadata contamination or borderline-date sources.

Run: python mine_leaked_knowledge.py
Requires: GPU (run on RunPod)
Output: results/leaked_candidates.json
"""

import json

import torch

import config
from model_loader import load_model


# Post-1930 facts to test, focusing on 1930-1945 borderline period
# plus some well-known later facts as negative controls
BORDERLINE_FACTS = [
    # 1930-1935 (most likely to leak through borderline dates)
    {"prompt": "The planet Pluto was discovered in the year", "expected": "1930", "year": 1930},
    {"prompt": "The Empire State Building was completed in", "expected": "1931", "year": 1931},
    {"prompt": "Franklin Roosevelt was first elected president in", "expected": "1932", "year": 1932},
    {"prompt": "Adolf Hitler became Chancellor of Germany in", "expected": "1933", "year": 1933},
    {"prompt": "The Dust Bowl devastated the American", "expected": "Great Plains", "year": 1934},
    {"prompt": "The Hoover Dam was completed in the year", "expected": "1935", "year": 1935},
    {"prompt": "Jesse Owens won four gold medals at the Olympics in", "expected": "Berlin", "year": 1936},
    {"prompt": "The Hindenburg disaster occurred in", "expected": "1937", "year": 1937},
    {"prompt": "Nylon was first commercially produced by", "expected": "DuPont", "year": 1938},
    {"prompt": "Germany invaded Poland in September", "expected": "1939", "year": 1939},
    {"prompt": "The Golden Gate Bridge is located in", "expected": "San Francisco", "year": 1937},
    {"prompt": "The New Deal was a series of programs by President", "expected": "Roosevelt", "year": 1933},
    {"prompt": "Amelia Earhart disappeared over the Pacific in", "expected": "1937", "year": 1937},
    {"prompt": "The Spanish Civil War began in", "expected": "1936", "year": 1936},
    {"prompt": "Penicillin was first used as a medicine by Alexander", "expected": "Fleming", "year": 1929},
    {"prompt": "The stock market crashed in October", "expected": "1929", "year": 1929},
    {"prompt": "The Great Depression began in the year", "expected": "1929", "year": 1929},
    {"prompt": "Mahatma Gandhi led the Salt March in", "expected": "1930", "year": 1930},
    {"prompt": "The Nazis held the Nuremberg rallies in the city of", "expected": "Nuremberg", "year": 1933},
    {"prompt": "Japan invaded Manchuria in", "expected": "1931", "year": 1931},

    # 1935-1945
    {"prompt": "World War II began in the year", "expected": "1939", "year": 1939},
    {"prompt": "The Battle of Britain was fought in", "expected": "1940", "year": 1940},
    {"prompt": "Japan attacked Pearl Harbor on December 7,", "expected": "1941", "year": 1941},
    {"prompt": "The Battle of Stalingrad took place in", "expected": "1942", "year": 1942},
    {"prompt": "D-Day, the Allied invasion of Normandy, occurred in", "expected": "1944", "year": 1944},
    {"prompt": "The United Nations was founded in", "expected": "1945", "year": 1945},
    {"prompt": "The atomic bomb was first tested at", "expected": "Trinity", "year": 1945},
    {"prompt": "Anne Frank wrote her diary while hiding in", "expected": "Amsterdam", "year": 1942},
    {"prompt": "The Manhattan Project developed the first", "expected": "atomic", "year": 1942},
    {"prompt": "Radar technology was crucial in the Battle of", "expected": "Britain", "year": 1940},

    # 1945-1960 (less likely to leak)
    {"prompt": "The Marshall Plan helped rebuild", "expected": "Europe", "year": 1948},
    {"prompt": "The State of Israel was established in", "expected": "1948", "year": 1948},
    {"prompt": "NATO was established in the year", "expected": "1949", "year": 1949},
    {"prompt": "The Korean War began in", "expected": "1950", "year": 1950},
    {"prompt": "Queen Elizabeth II became queen in", "expected": "1952", "year": 1952},
    {"prompt": "The double helix structure of DNA was discovered in", "expected": "1953", "year": 1953},
    {"prompt": "Rosa Parks refused to give up her seat in", "expected": "1955", "year": 1955},
    {"prompt": "Sputnik, the first artificial satellite, was launched by the", "expected": "Soviet", "year": 1957},
    {"prompt": "The European Economic Community was established in", "expected": "1957", "year": 1957},
    {"prompt": "Fidel Castro came to power in Cuba in", "expected": "1959", "year": 1959},

    # 1960-1990 (negative controls — should NOT be leaked)
    {"prompt": "The Cuban Missile Crisis occurred in", "expected": "1962", "year": 1962},
    {"prompt": "John F. Kennedy was assassinated in", "expected": "1963", "year": 1963},
    {"prompt": "The Civil Rights Act was signed in", "expected": "1964", "year": 1964},
    {"prompt": "The first heart transplant was performed by", "expected": "Barnard", "year": 1967},
    {"prompt": "Woodstock music festival took place in", "expected": "1969", "year": 1969},
    {"prompt": "The Watergate scandal involved President", "expected": "Nixon", "year": 1972},
    {"prompt": "The Vietnam War ended in", "expected": "1975", "year": 1975},
    {"prompt": "The Camp David Accords were signed by", "expected": "Carter", "year": 1978},
    {"prompt": "Margaret Thatcher became Prime Minister in", "expected": "1979", "year": 1979},
    {"prompt": "The AIDS epidemic was first identified in", "expected": "1981", "year": 1981},
    {"prompt": "The Falklands War was between Britain and", "expected": "Argentina", "year": 1982},
    {"prompt": "The Internet was originally developed by", "expected": "DARPA", "year": 1969},
    {"prompt": "The first personal computer was the", "expected": "Apple", "year": 1977},
    {"prompt": "Mikhail Gorbachev introduced the policy of", "expected": "perestroika", "year": 1986},
    {"prompt": "Nelson Mandela was released from prison in", "expected": "1990", "year": 1990},

    # Additional borderline 1928-1932
    {"prompt": "Alexander Fleming discovered penicillin in", "expected": "1928", "year": 1928},
    {"prompt": "The first Academy Awards ceremony was held in", "expected": "1929", "year": 1929},
    {"prompt": "The Chrysler Building was completed in New York in", "expected": "1930", "year": 1930},
    {"prompt": "The Star-Spangled Banner became the national anthem in", "expected": "1931", "year": 1931},
    {"prompt": "Aldous Huxley published Brave New World in", "expected": "1932", "year": 1932},
    {"prompt": "The first FIFA World Cup was held in", "expected": "Uruguay", "year": 1930},
    {"prompt": "The Smoot-Hawley Tariff Act was signed in", "expected": "1930", "year": 1930},

    # Science near boundary
    {"prompt": "Edwin Hubble showed that the universe is", "expected": "expanding", "year": 1929},
    {"prompt": "The neutron was discovered by James", "expected": "Chadwick", "year": 1932},
    {"prompt": "Dirac predicted the existence of the", "expected": "positron", "year": 1931},
    {"prompt": "Kurt Godel published his incompleteness theorems in", "expected": "1931", "year": 1931},
    {"prompt": "Heavy water was discovered in", "expected": "1932", "year": 1932},

    # Culture near boundary
    {"prompt": "The first talking motion picture was The Jazz", "expected": "Singer", "year": 1927},
    {"prompt": "Mickey Mouse first appeared in the cartoon Steamboat", "expected": "Willie", "year": 1928},
    {"prompt": "Gone with the Wind was published by Margaret", "expected": "Mitchell", "year": 1936},
    {"prompt": "Walt Disney released the first full-length animated film Snow White in", "expected": "1937", "year": 1937},
]


def mine_leaked():
    print("=" * 60)
    print("MINING FOR LEAKED KNOWLEDGE IN TALKIE-1930")
    print("=" * 60)

    model = load_model(config.VINTAGE_MODEL_ID)

    results = []
    correct_by_decade = {}

    for item in BORDERLINE_FACTS:
        completion = model.generate(
            item["prompt"], max_new_tokens=30, temperature=0.0
        )
        generated = completion.strip()
        hit = item["expected"].lower() in generated.lower()
        mark = "Y" if hit else "N"

        decade = (item["year"] // 10) * 10
        if decade not in correct_by_decade:
            correct_by_decade[decade] = {"correct": 0, "total": 0}
        correct_by_decade[decade]["total"] += 1
        if hit:
            correct_by_decade[decade]["correct"] += 1

        result = {
            "prompt": item["prompt"],
            "expected": item["expected"],
            "year": item["year"],
            "completion": generated[:100],
            "correct": hit,
        }
        results.append(result)
        print(f"  [{mark}] ({item['year']}) \"{item['prompt']}\"")
        print(f"       -> \"{generated[:70]}\"")

    del model
    if config.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY BY DECADE")
    print(f"{'='*60}")
    for decade in sorted(correct_by_decade.keys()):
        d = correct_by_decade[decade]
        pct = d["correct"] / d["total"] * 100 if d["total"] > 0 else 0
        print(f"  {decade}s: {d['correct']}/{d['total']} ({pct:.0f}%)")

    leaked_candidates = [r for r in results if r["correct"]]
    print(f"\n  Total leaked candidates: {len(leaked_candidates)}")
    print(f"  Total tested: {len(results)}")

    # Save
    out_path = config.RESULTS_DIR / "leaked_candidates.json"
    with open(out_path, "w") as f:
        json.dump({
            "all_results": results,
            "leaked_candidates": leaked_candidates,
            "by_decade": {str(k): v for k, v in correct_by_decade.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    print(f"\n  NEXT STEP: Add leaked candidates to data/temporal_facts.json")
    print(f"  under the 'leaked_knowledge' category, then re-run experiment3.")


if __name__ == "__main__":
    mine_leaked()
