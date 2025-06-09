import pandas as pd

# 1. read metrics.csv
metrics_path = "../logs/p97_l2_s0_div/lightning_logs/version_12/metrics.csv"
df = pd.read_csv(metrics_path)

# 2. find 1st time train_acc > 99.5  step
train_cross = df[df["train_acc"] > 99.5]
if train_cross.empty:
    raise ValueError("The training accuracy has never exceeded 99.5%.")
a = int(train_cross.iloc[0]["step"])

# 3. find 1st time val_acc > 99.5   step
val_cross = df[df["val_acc"] > 99.5]
if val_cross.empty:
    raise ValueError("The testing accuracy has never exceeded 99.5%.")
b = int(val_cross.iloc[0]["step"])

# 4. calculate difference
print(f"1st train_acc > 99.5%, step a = {a}")
print(f"1st val_acc   > 99.5%, step b = {b}")
print(f"b - a = {b - a}")
