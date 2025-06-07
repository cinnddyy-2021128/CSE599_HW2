# CSE599 HW2
_To run question 1_

---

## 📁 Directory Overview

| Path / File | Purpose |
|-------------|---------|
| `model.py` | Core model architecture and helper functions |
| `train.py` | Script version of the training pipeline |
| `train.ipynb` | Jupyter notebook that **wraps** `train.py` via `%run` for interactive experiments |
| `inference.ipynb` | Notebook for quick, interactive inference / demo |
| `tiny.txt` | Dataset used in docs & tests |
| `logs/` | Auto‑generated training logs |
| `Question1_discussion.txt` | Described modifications made to the original codebase, and any challenges faced. |

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


