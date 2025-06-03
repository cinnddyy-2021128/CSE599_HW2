import os
import random
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# 1. Function: Generate modular arithmetic data (add, sub, div)
# ----------------------------------------------------------------------
def generate_modular_data(p, operation):
    data = []
    for a in range(p):
        for b in range(p):
            if operation == "add":
                c = (a + b) % p
                data.append(f"{a}+{b}={c}")
            elif operation == "sub":
                c = (a - b) % p
                data.append(f"{a}-{b}={c}")
            elif operation == "div":
                if b == 0:
                    continue
                inv_b = pow(b, p - 2, p)  # modular inverse since p is prime
                c = (a * inv_b) % p
                data.append(f"{a}/{b}={c}")
            else:
                raise ValueError("Invalid operation. Choose from 'add', 'sub', 'div'.")
    return data

# ----------------------------------------------------------------------
# 2. Main script: generate and split using scikit-learn
# ----------------------------------------------------------------------
def main():
    # Fix random seed for reproducibility
    SEED = 42

    # Directory to store generated text files
    os.makedirs("modular_data", exist_ok=True)

    # Primes and operations
    primes = [97, 113]
    operations = ["add", "sub", "div"]

    for p in primes:
        for op in operations:
            # 2.1 Generate all equations for this (p, op)
            data = generate_modular_data(p, op)

            # 2.2 First split: 10% test, 90% train+val
            train_val, test = train_test_split(
                data,
                test_size=0.20,
                random_state=SEED,
                shuffle=True
            )

        
            for split_name, split_list in zip(["train_val", "test"], [train_val, test]):
                filename = f"modular_data/{op}_{p}_{split_name}.txt"
                with open(filename, "w") as f:
                    for line in split_list:
                        f.write(line + "\n")

            n_train_val = len(train_val)
            n_test = len(test)
            print(f"p = {p}, op = {op:3s} → train: {n_train_val}, test: {n_test}")

if __name__ == "__main__":
    main()
