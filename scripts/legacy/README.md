# Legacy training scripts (pre-runner.py)

Superseded by `runner.py` + `config.jsonc`. Kept for reference and backward compatibility.

```bash
python scripts/legacy/train_conditional_unet.py --epochs 10 --device cuda
python scripts/legacy/eval_conditional_unet.py --checkpoint runs/conditional_unet/best.pt
```

Run from the repo root (`A:\MANGA`).
