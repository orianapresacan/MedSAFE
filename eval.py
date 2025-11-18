import os
import json

ROOT = r"outputs"   

print("\nCollecting abstention metrics from all logs...\n")

for folder, subdirs, files in os.walk(ROOT):
    if "logs" not in folder.lower():
        continue

    for filename in files:
        if not filename.lower().endswith(".json"):
            continue

        path = os.path.join(folder, filename)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue

        samples = data.get("samples", [])
        if not samples:
            print(f"No samples in {path}")
            continue

        sample = samples[0]  

        aj = sample.get("scores", {}).get("alignment_judge", {})
        scores = aj.get("value", aj)

        print("=" * 60)
        print(f"File: {path}")
        print("Abstention metrics:\n")

        for key, value in scores.items():
            print(f"{key}: {value}")

        print()  
