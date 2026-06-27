#!/bin/bash
echo "Waiting for train_fno.py to finish..."
while pgrep -f "train_fno.py" > /dev/null; do
    sleep 30
done
echo "FNO training finished. Starting PNO training..."
PYTHONUNBUFFERED=1 .venv/bin/python3 train_pno.py \
  --data_dir data/data_new_10k_clean \
  --cache data/cache_10k_clean/pno_cache.npz \
  --fno_checkpoint checkpoints/fno_10k_clean/model_best.ckpt \
  --fno_config checkpoints/fno_10k_clean/model_config.json \
  --output_dir checkpoints/pno_10k_clean \
  --epochs 100 \
  --batch_size 32 \
  > /tmp/train_pno_10k_clean.log 2>&1
echo "PNO training finished."
