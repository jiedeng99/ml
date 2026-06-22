import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm import tqdm

from self_attention.data import get_inference_dataloader
from self_attention.model import Classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Run speaker classifier inference.")
    parser.add_argument("--data-dir", default="./Dataset")
    parser.add_argument("--model-path", default="./model.ckpt")
    parser.add_argument("--output-path", default="./output.csv")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-workers", type=int, default=8)
    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device}")

    mapping_path = Path(args.data_dir) / "mapping.json"
    with mapping_path.open() as f:
        mapping = json.load(f)

    dataloader = get_inference_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
    )
    print("[Info]: Finish loading data", flush=True)

    speaker_num = len(mapping["id2speaker"])
    model = Classifier(n_spks=speaker_num).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("[Info]: Finish loading model", flush=True)

    results = [["id", "Category"]]
    for feat_paths, mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open(args.output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)


if __name__ == "__main__":
    main(parse_args())
