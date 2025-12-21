from datasets import load_dataset
import json

dataset = load_dataset("tau/commonsense_qa")

# Export the validation (or dev) split
with open("dev_rand_split.jsonl", "w") as f:
    for item in dataset["validation"]:
        f.write(json.dumps(item) + "\n")

print("✅ Saved CommonsenseQA dev split to dev_rand_split.jsonl")
