# CSE599 HW2
# To run question 1

---

## 📁 Directory Overview

| Path / File | Purpose |
|-------------|---------|
| `model.py` | Core model architecture and helper functions |
| `train.py` | Script version of the training pipeline for question 1|
| `train.ipynb` | Jupyter notebook that **wraps** `train.py` via `%run` for interactive experiments |
| `inference.ipynb` | Notebook for quick, interactive inference / demo |
| `tiny.txt` | Dataset used in docs & tests |
| `logs/` | Auto‑generated training logs |
| `training.py` | Script version of the training pipeline for question 2 |

---

## 🛠 Setup

1. **Clone the repo**

   ```bash
   git clone <https://github.com/cinnddyy-2021128/CSE599_HW2.git>
   cd <CSE599_HW2>

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate

3. **Install dependencies**

   ```bash
   pip install torch
   pip install numpy pandas tqdm jupyter matplotlib

## 🚂 Training

1. **Launch Jupyter:**

   ```bash
   jupyter notebook train.ipynb

2. **Open `train.ipynb`, edit the first cell if you want to tweak hyper‑parameters, then Run All.**

3. **Checkpoints and metrics appear under `logs`.**

## 🔍 Inference / Demo

1. **Open `inference.ipynb`, select a saved model checkpoint, and run the cells.**

2. **Change the prompt here `prompt = 'I l'`**

 # To run question 2

# OpenAI Grok Curve Experiments

# 2.1 Data Generation
`scripts/make_data.py`

# 2.2

`python scripts/train.py`

`
--n_layers 2
--n_heads 4
--d_model
128
--random_seed
0
--datadir
"../data97"
--math_operator
"/"
--logdir
../logs/p97_l2_s0_div
--max_steps
100000
--max_epochs
10000
`


