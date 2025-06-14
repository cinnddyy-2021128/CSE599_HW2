# CSE599 HW2

| File/folder | Purpose |
|-------------|---------|
| `HW2.pdf` | All the plots and free response questions organized|
| `1_deliverables` | folder including log, ckpt, train and inference code for question 1|
| `2_1_deliverables` | Generated train/test splits for question 2.1|
| `2_2_deliverables` | Final model checkpoint for one of the seeds for question 2.2|
| `2_3_deliverables` | Final model ckpt for one seed for question 2.3|
| `Q1` | Every file for question 1|
| `Q2` | Every file for question 2|

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
`python scripts/make_data.py`

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


