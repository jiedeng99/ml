import argparse
import copy

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from self_attention.data import get_dataloader
from self_attention.model import Classifier
from self_attention.schedule import get_cosine_schedule_with_warmup


def parse_args():
    parser = argparse.ArgumentParser(description="Train a self-attention speaker classifier.")
    parser.add_argument("--data-dir", default="./Dataset")
    parser.add_argument("--save-path", default="./model.ckpt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--valid-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=4000)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--segment-len", type=int, default=128)
    return parser.parse_args()


def model_fn(batch, model, criterion, device):
    mels, labels = batch
    mels, labels = mels.to(device), labels.to(device)

    outs = model(mels)
    loss = criterion(outs, labels)
    preds = outs.argmax(1)
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy


def validate(dataloader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

        pbar.update(len(batch[0]))
        pbar.set_postfix(
            loss=f"{running_loss / (i + 1):.2f}",
            accuracy=f"{running_accuracy / (i + 1):.2f}",
        )

    pbar.close()
    model.train()
    return running_accuracy / len(dataloader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device}")

    train_loader, valid_loader, speaker_num = get_dataloader(
        args.data_dir,
        args.batch_size,
        args.n_workers,
        segment_len=args.segment_len,
    )
    train_iterator = iter(train_loader)
    print("[Info]: Dataloader Done", flush=True)

    model = Classifier(n_spks=speaker_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        args.warmup_steps,
        args.total_steps,
    )
    print("[Info]: Model, Criterion, Optimizer, Scheduler Done", flush=True)

    best_accuracy = -1.0
    best_state_dict = None
    pbar = tqdm(total=args.valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(args.total_steps):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step + 1,
        )

        if (step + 1) % args.valid_steps == 0:
            pbar.close()
            valid_accuracy = validate(valid_loader, model, criterion, device)
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = copy.deepcopy(model.state_dict())

            pbar = tqdm(total=args.valid_steps, ncols=0, desc="Train", unit=" step")

        if (step + 1) % args.save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, args.save_path)
            pbar.write(
                f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})"
            )

    pbar.close()

    if best_state_dict is not None:
        torch.save(best_state_dict, args.save_path)


if __name__ == "__main__":
    main(parse_args())
