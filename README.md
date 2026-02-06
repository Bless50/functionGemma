# functionGemma

This repository currently contains a Colab notebook for fine-tuning **FunctionGemma 270M** on the **Mobile Actions** dataset and converting the resulting model to `.litertlm` for on-device deployment.

- Notebook: `FunctionGemma/[FunctionGemma]Finetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb`

---

## What likely happened in your run

From your description, you probably hit one (or more) common issues:

1. **Runtime reset / disconnection in Colab** before artifacts were persisted.
2. **Checkpoint path confusion** (`/content/...` is ephemeral unless copied out).
3. **LoRA config + prompt format mismatch**, causing poor or empty outputs.
4. **Wrong checkpoint selected for conversion** (converting base or partial output instead of final tuned checkpoint).

---

## Recovery plan (Google Drive + Colab)

Use this sequence to recover exactly what you trained:

1. **Mount Drive** and inspect all likely output folders.
2. Find files such as:
   - `trainer_state.json`
   - `config.json`
   - adapter files (`adapter_model.safetensors`, `adapter_config.json`) if LoRA was used
   - tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`)
   - model shards (`pytorch_model*.bin` or safetensors)
3. Confirm the checkpoint that has the latest global step in `trainer_state.json`.
4. Re-load that checkpoint in Colab and run a quick inference sanity test.
5. Only then run `.litertlm` conversion.

### Quick Colab snippet to locate candidate checkpoints

```python
from pathlib import Path

roots = [
    Path('/content/drive/MyDrive'),
    Path('/content'),
]
markers = {
    'trainer_state.json',
    'adapter_model.safetensors',
    'adapter_config.json',
    'config.json',
}

candidates = []
for root in roots:
    if not root.exists():
        continue
    for p in root.rglob('*'):
        if p.name in markers:
            candidates.append(str(p.parent))

for d in sorted(set(candidates)):
    print(d)
```

---

## LoRA troubleshooting checklist (when output quality collapses)

If you changed tuning method (LoRA/QLoRA/etc.) and outputs became unusable:

- Verify your **training text template == inference prompt template**.
- Ensure **EOS token handling** is correct and labels aren’t accidentally masked to nothing.
- Start conservative:
  - `lora_r`: 8 or 16
  - `lora_alpha`: 16 or 32
  - `lora_dropout`: `0.05`
- Keep eval samples fixed and compare:
  - base model output
  - current fine-tuned output
  - previous known-good checkpoint output
- If the model returns repetitive or blank behavior, reduce LR and/or train for fewer steps, then re-evaluate.

---

## Minimal “known-good” workflow to avoid losing work

1. Save outputs to a unique run directory, e.g. `/content/mobile-actions-functiongemma/run-YYYYMMDD-HHMM`.
2. At end of every N steps/epoch, copy critical files to Drive.
3. Save:
   - training args/config
   - tokenizer
   - checkpoint
   - a small eval report (inputs + outputs)
4. Convert to `.litertlm` only after sanity checks pass.

---

## Suggested next step

If you want, I can add a small Python utility in this repo that:

- scans a mounted Drive folder,
- ranks candidate checkpoints by readiness,
- validates required files for conversion,
- and emits a single “best checkpoint path” to use in Colab.

That would make recovery from interrupted Colab sessions much easier.
