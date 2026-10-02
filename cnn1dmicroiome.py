"""
DNA 1D-CNN classifier with a Click CLI.

Uses a 1D CNN with AdaptiveAvgPool1d, so it accepts sequences of any length
at inference time (as long as they're long enough to survive two kernel-7
convolutions, i.e. length >= 13). Training sequences are padded to a common
length only so they can be batched together.

Usage
-----
Train:
    python dna_cnn_cli.py train \\
        --data sequences.csv \\
        --seq-col sequences --label-col label \\
        --epochs 5 --lr 1e-3 --bs 64 \\
        --model-out dna_cnn_model.pth

Predict:
    python dna_cnn_cli.py predict \\
        --model dna_cnn_model.pth \\
        --data new_sequences.csv --seq-col sequences \\
        --out predictions.csv
"""

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from fastai.learner import Learner
from fastai.losses import BCEWithLogitsLossFlat
from fastai.metrics import accuracy, RocAucBinary
from fastai.data.core import DataLoaders

BASE_MAPPING = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}

MIN_SEQ_LEN = 13  # two Conv1d(kernel_size=7) layers, no padding


def one_hot_encode(sequence: str) -> np.ndarray:
    """One-hot encode a DNA sequence, treating unknown bases as N."""
    return np.array(
        [BASE_MAPPING.get(base.upper(), BASE_MAPPING["N"]) for base in sequence],
        dtype=np.float32,
    )


def pad_sequences(sequences, pad_char="N"):
    max_len = max(len(s) for s in sequences)
    max_len = max(max_len, MIN_SEQ_LEN)
    return [s + pad_char * (max_len - len(s)) for s in sequences], max_len


class DNACNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


def build_dataloaders(X, y, bs, valid_pct, seed):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=valid_pct, random_state=seed, stratify=y
    )
    X_train = torch.tensor(X_train).permute(0, 2, 1)
    X_valid = torch.tensor(X_valid).permute(0, 2, 1)
    y_train = torch.tensor(y_train)
    y_valid = torch.tensor(y_valid)

    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    return DataLoaders.from_dsets(train_ds, valid_ds, bs=bs)


@click.group()
def cli():
    """Train and run a 1D-CNN DNA sequence classifier."""
    pass


@cli.command()
@click.option("--data", "data_path", required=True, type=click.Path(exists=True),
              help="CSV file with sequence and label columns.")
@click.option("--seq-col", default="sequences", show_default=True, help="Column name holding DNA sequences.")
@click.option("--label-col", default="label", show_default=True, help="Column name holding binary labels.")
@click.option("--epochs", default=5, show_default=True, type=int)
@click.option("--lr", default=1e-3, show_default=True, type=float)
@click.option("--bs", default=64, show_default=True, type=int, help="Batch size.")
@click.option("--valid-pct", default=0.2, show_default=True, type=float, help="Fraction held out for validation.")
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--model-out", default="dna_cnn_model.pth", show_default=True,
              type=click.Path(), help="Where to save the trained model's state_dict.")
@click.option("--plot-loss/--no-plot-loss", default=False, help="Show the training loss plot.")
def train(data_path, seq_col, label_col, epochs, lr, bs, valid_pct, seed, model_out, plot_loss):
    """Train the DNA CNN on labeled sequence data."""
    df = pd.read_csv(data_path)
    sequences = df[seq_col].astype(str).to_list()
    labels = df[label_col].to_list()

    padded, max_len = pad_sequences(sequences)
    click.echo(f"Padded {len(sequences)} sequences to length {max_len}.")

    X = np.array([one_hot_encode(s) for s in padded], dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    dls = build_dataloaders(X, y, bs=bs, valid_pct=valid_pct, seed=seed)

    model = DNACNN()
    learner = Learner(
        dls, model, loss_func=BCEWithLogitsLossFlat(), metrics=[accuracy, RocAucBinary()]
    )

    learner.fit(epochs, lr=lr)
    click.echo("Validation results (loss, " + ", ".join(m.name for m in learner.metrics) + "):")
    click.echo(str(learner.validate()))

    if plot_loss:
        learner.recorder.plot_loss()

    preds, targets = learner.get_preds(dl=learner.dls.valid)
    probabilities = torch.sigmoid(preds.squeeze())
    click.echo(f"Sample predicted probabilities: {probabilities[:10].tolist()}")
    click.echo(f"Sample true labels:            {targets[:10].squeeze().numpy().tolist()}")

    torch.save(learner.model.state_dict(), model_out)
    click.echo(f"Model saved to {model_out}")


@cli.command()
@click.option("--model", "model_path", required=True, type=click.Path(exists=True),
              help="Path to a saved model state_dict (.pth).")
@click.option("--data", "data_path", required=True, type=click.Path(exists=True),
              help="CSV file with a column of sequences to score.")
@click.option("--seq-col", default="sequences", show_default=True, help="Column name holding DNA sequences.")
@click.option("--threshold", default=0.5, show_default=True, type=float, help="Decision threshold for label 1.")
@click.option("--out", "out_path", default="predictions.csv", show_default=True,
              type=click.Path(), help="Where to write the predictions CSV.")
def predict(model_path, data_path, seq_col, threshold, out_path):
    """Score new sequences with a trained model."""
    df = pd.read_csv(data_path)
    sequences = df[seq_col].astype(str).to_list()

    model = DNACNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    predictions = []
    with torch.no_grad():
        for seq in sequences:
            if len(seq) < MIN_SEQ_LEN:
                seq = seq + "N" * (MIN_SEQ_LEN - len(seq))
            encoded = one_hot_encode(seq)
            xb = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
            logits = model(xb)
            probability = torch.sigmoid(logits.squeeze()).item()
            predictions.append(probability)

    probabilities = np.array(predictions)
    predicted_labels = (probabilities >= threshold).astype(int)

    results = pd.DataFrame({
        "sequence": sequences,
        "probability": probabilities,
        "prediction": predicted_labels,
    })
    results.to_csv(out_path, index=False)
    click.echo(results)
    click.echo(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    cli()