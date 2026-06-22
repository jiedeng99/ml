# Self-Attention Speaker Classifier

This project is a Python package version of `notebooks/SelfAttention.ipynb`. It trains a transformer-encoder classifier on mel-spectrogram features and can generate a submission CSV for inference data.

## Layout

- `self_attention/data.py`: training and inference datasets plus dataloaders
- `self_attention/model.py`: transformer encoder classifier
- `self_attention/schedule.py`: cosine schedule with warmup
- `self_attention/train.py`: training entrypoint
- `self_attention/infer.py`: inference entrypoint
- `notebooks/SelfAttention.ipynb`: original notebook
- `notes/SelfAttention.md`: learning notes

## Setup

Install dependencies from this directory:

```bash
pip install -r requirements.txt
```

The notebook expected a dataset directory with:

- `mapping.json`
- `metadata.json`
- `testdata.json` for inference
- feature files referenced by the JSON metadata

By default, commands look for `./Dataset` and save `./model.ckpt`.

## Train

```bash
python -m self_attention.train --data-dir ./Dataset --save-path ./model.ckpt
```

Useful options:

```bash
python -m self_attention.train \
  --data-dir ./Dataset \
  --save-path ./model.ckpt \
  --batch-size 32 \
  --n-workers 8 \
  --valid-steps 2000 \
  --warmup-steps 1000 \
  --save-steps 4000 \
  --total-steps 5000
```

## Inference

```bash
python -m self_attention.infer \
  --data-dir ./Dataset \
  --model-path ./model.ckpt \
  --output-path ./output.csv
```

The output CSV contains `id,Category`, matching the original notebook.
