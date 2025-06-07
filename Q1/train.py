import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from model import GPT, GPTConfig          # starter file supplied by HW

# ----------------------------------------------------------------------
# 1. Tokenizer and dataset
# ----------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, text: str):
        self.stoi = {ch: i for i, ch in enumerate(sorted(set(text)))} # char -> id
        self.itos = {i: ch for ch, i in self.stoi.items()} # id -> char

    @property
    def vocab_size(self):               
        return len(self.stoi)

    def encode(self, s: str):
        return [self.stoi[c] for c in s] # str -> list of ids

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids) # list of ids -> str


class CharDataset(Dataset):
    def __init__(self, ids, block_size: int):
        self.data  = ids
        self.block = block_size

    def __len__(self):
        return len(self.data) - self.block

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ----------------------------------------------------------------------
# 2. Utility helpers
# ----------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def masked_ce_loss(logits, targets, mask=None):
    # logits: (B, T, V), targets: (B, T), mask: (B, T)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction='none'
    )
    if mask is not None:
        loss = loss * mask.view(-1)
        return loss.sum() / mask.sum()
    else:
        return loss.mean()


def pad_collate(batch, pad_id):
    xs, ys = zip(*batch)                          # tuples of tensors
    T_max  = max(x.size(0) for x in xs)

    def _pad(seq, fill):
        if seq.size(0) == T_max:
            return seq
        pad_len = T_max - seq.size(0)
        return torch.cat([seq, torch.full((pad_len,), fill, dtype=seq.dtype)])

    xs_pad = torch.stack([_pad(x, pad_id) for x in xs])  # (B, T_max)
    ys_pad = torch.stack([_pad(y, pad_id) for y in ys])
    mask   = (xs_pad != pad_id).to(torch.float32)        # float mask
    return xs_pad, ys_pad, mask


# ----------------------------------------------------------------------
# 3. Argument parser
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HW2 Transformer trainer")

    # I/O -------------------------------------------------------------------
    p.add_argument('--data_path',   type=Path, required=True,
                   help='Plain-text corpus')
    p.add_argument('--logdir',      type=Path, default=Path('logs'))

    # Model --------------------------------------------------------------
    p.add_argument('--n_layer',     type=int,   default=2)
    p.add_argument('--n_head',      type=int,   default=4)
    p.add_argument('--n_embd',      type=int,   default=128)
    p.add_argument('--block_size',  type=int,   default=16)

    # Optimization -------------------------------------------------------
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--betas',        type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument('--steps',        type=int,   default=2_000)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--save_every',   type=int,   default=500)
    return p


# ----------------------------------------------------------------------
# 4. Main training loop
# ----------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    args.logdir.mkdir(parents=True, exist_ok=True)

    # ---- save config for reproducibility ---------------------------------
    (args.logdir / 'config.json').write_text(
        json.dumps(vars(args), indent=2, default=str))

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- load & tokenize corpus -----------------------------------------
    text = args.data_path.read_text(encoding='utf-8')
    tok  = CharTokenizer(text)
    ids  = tok.encode(text)
    pad_id = tok.vocab_size            # reserve *next* id for <pad>

    # ---- build dataset & dataloader --------------------------------------
    ds = CharDataset(ids, args.block_size)
    dl = DataLoader(ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=lambda b: pad_collate(b, pad_id))

    # ---- instantiate model ----------------------------------------------
    conf = GPTConfig(
        block_size=args.block_size,
        vocab_size=tok.vocab_size + 1,   # +1 for <pad>
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
    )
    model = GPT(conf).to(device)
    optim = model.configure_optimizers(
        weight_decay=args.weight_decay, learning_rate=args.lr, betas=args.betas,
        device_type=device)

    # ---- training --------------------------------------------------------
    model.train()
    t0 = time.time()
    step = 0
    while step < args.steps:
        for x, y, mask in dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            mask[:, :3] = 0
            pad_mask = (x != pad_id)
            logits = model(x, pad_mask=pad_mask)                           # (B, T, V)
            loss   = masked_ce_loss(logits, y, mask)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            step += 1
            if step % 50 == 0:
                print(f"[{step:>6}/{args.steps}] loss {loss.item():.4f}")

            if step % args.save_every == 0 or step == args.steps:
                ckpt_path = args.logdir / f"ckpt_{step:07d}.pt"
                torch.save({
                    'model':     model.state_dict(),
                    'gpt_conf':  conf.__dict__,
                    'tok':       tok.stoi,
                    'step':      step,
                }, ckpt_path)
                print("⇪ saved", ckpt_path)

            if step >= args.steps:
                break

    print(f"Done in {(time.time()-t0)/60:.1f} min. Final loss {loss.item():.4f}")


if __name__ == "__main__":            
    main()
