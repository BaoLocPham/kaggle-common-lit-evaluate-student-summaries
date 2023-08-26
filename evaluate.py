from pathlib import Path
import json
import numpy as np

scores = {}
preds = {}

# model name, batch size, max seq len
model_configs = [
    ('roberta-base', 10, 32),
    ('google/electra-base-discriminator', 10, 32),
    ('microsoft/deberta-base-mnli', 6, 32),
]

for model, _, _ in model_configs:
    scores[model] = []
    for fold in range(1):

        output = f"/kaggle/working/{model.split('/')[-1]}_fold{fold}"
        p = Path(output) / "config.json"

        with open(p) as fp:
            cfg = json.load(fp)

        print(
            f"MCRMSE for model {model}, fold {fold}: {cfg['best_metric']:.4f}")
        scores[model].append(cfg['best_metric'])

    print()

cv = []
for model, _, _ in model_configs:
    cv.append(sum(scores[model])/len(scores[model]))
    print(f"CV (model {model}): {cv[-1]}")


print(f"Average CV across model: {sum(cv)/len(cv)}")