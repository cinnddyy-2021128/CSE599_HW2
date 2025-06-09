#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    metrics_csv = "../logs/p97_l2_s0_div/lightning_logs/version_6/metrics.csv"
    out_dir = "../logs/p97_l2_s0_div/plots_simple"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(metrics_csv)


    train_df = df[df["train_loss"].notna()].copy()
    val_df   = df[df["val_loss"].notna()].copy()

    plt.figure(figsize=(6,4))
    plt.plot(
        train_df["epoch"], train_df["train_loss"],
        label="Train Loss",
        linewidth=2,
    )
    plt.plot(
        val_df["epoch"], val_df["val_loss"],
        label="Test Loss",
        linewidth=2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Test Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/loss_curve.png")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(
        train_df["epoch"], train_df["train_acc"],
        label="Train Accuracy",
        linewidth=2,
    )
    plt.plot(
        val_df["epoch"], val_df["val_acc"],
        label="Test Accuracy",
        linewidth=2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train vs Test Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/accuracy_curve.png")
    plt.close()

    print("Saved figures:")
    print(f"  - {out_dir}/loss_curve.png")
    print(f"  - {out_dir}/accuracy_curve.png")
