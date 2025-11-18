import json

p = r"outputs\med_abs_lab_interpretation_ambiguous\logs\2025-11-18T10-45-51+02-00_task_MKAhHtETmAdtdjovMansk6.json"

with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)

sample = data["samples"][0]


aj = sample.get("scores", {}).get("alignment_judge", {})
scores = aj.get("value", aj)   

print("All abstention metrics:\n")
for key, value in scores.items():
    print(f"{key}: {value}")

