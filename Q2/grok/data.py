import itertools
import math
import os
import sys
import random

import torch
from torch import Tensor, LongTensor
import numpy as np
from typing import Tuple, List, Dict, Any, Union, Optional
from tqdm import tqdm

from sympy.combinatorics.permutations import Permutation
from mod import Mod

import blobfile as bf


VALID_OPERATORS = {
    "+": "addition",
    "-": "subtraction",
    "*": "muliplication",
    "/": "division",
}
EOS_TOKEN = "<|eos|>"
EQ_TOKEN = "="
ODULUS = 97

MODULI = [113]
NUMS = list(range(max(MODULI)))
DEFAULT_DATA_DIR = "data"


def render(operand, join_str=""):
    if (
        isinstance(operand, list)
        or isinstance(operand, tuple)
        or isinstance(operand, np.ndarray)
    ):
        return join_str.join(map(render, operand))
    elif isinstance(operand, Permutation):
        return "".join(map(str, operand.array_form))
    elif isinstance(operand, Mod):
        return str(operand._value)
    else:
        return str(operand)


def create_data_files(data_dir: str = DEFAULT_DATA_DIR):
    ArithmeticTokenizer.create_token_file(data_dir)
    ArithmeticDataset.create_dataset_files(data_dir)


class ArithmeticTokenizer:
    """Stores the list of token text to token id mappings and converts between them"""

    token_file = "tokens.txt"

    def __init__(self, data_dir=DEFAULT_DATA_DIR) -> None:
        self.token_file = bf.join(data_dir, self.token_file)

        self.itos = self.get_tokens()

        self.stoi: Dict[str, int] = dict([(s, i) for i, s in enumerate(self.itos)])

    def _encode(self, s: str) -> Tensor:
        return LongTensor([self.stoi[t] for t in s.split(" ")])

    def encode(self, obj: Union[str, List]) -> Tensor:
        """
        Convert a string of text into a rank-1 tensor of token ids
        or convert a list of strings of text into a rank-2 tensor of token ids

        :param obj: the string or list of strings to convert
        :returns: a tensor of the token ids
        """
        if isinstance(obj, str):
            return self._encode(obj)
        elif isinstance(obj, list):
            return torch.stack([self._encode(s) for s in obj], dim=0)
        else:
            raise NotImplementedError

    def decode(self, tensor: Tensor, with_brackets: bool = False) -> str:
        """
        Convert a tensor of token ids into a string of text

        :param tensor: a tensor of the token ids
        :param with_brackets: if true, the returned string will include <> brackets
                              around the text corresponding to each token.
        :returns: string of these tokens.
        """
        indices = tensor.long()
        if with_brackets:
            l = "<"
            r = ">"
        else:
            l = ""
            r = ""
        tokens = [l + self.itos[i] + r for i in indices]
        return " ".join(tokens)

    def __len__(self) -> int:
        """
        :returns: the number of tokens in this vocabulary
        """
        return len(self.itos)

    '''
    @classmethod
    def get_tokens(cls):
        tokens = (
            [EOS_TOKEN, EQ_TOKEN]
            + list(sorted(list(VALID_OPERATORS.keys())))
            + list(map(render, NUMS))
            + list(map(render, itertools.permutations(range(5))))  # s5
        )
        return tokens
    '''

    @classmethod
    def get_tokens(cls):
        return (
                [ EQ_TOKEN, "+", "-", "/"]
                + [str(i) for i in range(max(MODULI))]
        )

    @classmethod
    def create_token_file(cls, data_dir: str):
        """
        Write each token from get_tokens() into data_dir/tokens.txt.

        :param data_dir: directory where tokens.txt will be created
        """
        # Ensure the data directory exists
        os.makedirs(data_dir, exist_ok=True)
        # Retrieve all tokens
        tokens = cls.get_tokens()
        # Construct the full path for tokens.txt
        token_path = bf.join(data_dir, cls.token_file)
        # Write each token on its own line
        with open(token_path, "w", encoding="utf-8") as f:
            for t in tokens:
                f.write(t + "\n")

class ArithmeticDataset:
    """A Dataset of arithmetic equations"""


    @classmethod
    def create_dataset_files(cls, data_dir: str):
        """
        Generate train/val/test splits for "+", "-", "/" and write to files.
        """
        # Store data_dir so that make_data can know where to write output files
        cls._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Only generate splits for the three arithmetic operators
        for operator in ["+", "-", "/"]:
            # Call make_data to generate and write train/val/test files
            cls.make_data(operator, operands=None, shuffle=True, seed=42)


    @classmethod
    def splits(
            cls,
            train_pct: float,
            operator: str,
            operand_length: Optional[int] = None,
            data_dir: str = DEFAULT_DATA_DIR,
    ):
        """
        Creates training and validation datasets by reading pre-generated
        "<ds_base>_train.txt" and "<ds_base>_val.txt" files under data_dir.

        :param train_pct: unused here, since splits already done by create_dataset_files
        :param operator: one of "+", "-", "/"
        :param operand_length: ignored for this task
        :param data_dir: directory where "<ds_base>_train.txt" etc. live
        :returns: (train_dataset, validation_dataset)
        """
        # Determine base name from operator, e.g. "+" -> "addition"
        ds_base = VALID_OPERATORS[operator]
        # Paths for train and val
        train_path = bf.join(data_dir, f"{ds_base}_train.txt")
        val_path = bf.join(data_dir, f"{ds_base}_val.txt")

        # Read all lines from train file (each line is "<|eos|> a o b = c mod p <|eos|>")
        with open(train_path, "r", encoding="utf-8") as f_tr:
            train_lines = [line.strip() for line in f_tr.readlines() if line.strip()]

        with open(val_path, "r", encoding="utf-8") as f_v:
            val_lines = [line.strip() for line in f_v.readlines() if line.strip()]

        # Construct ArithmeticDataset instances
        train_ds = cls(ds_base + "_train", train_lines, train=True, data_dir=data_dir)
        val_ds = cls(ds_base + "_val", val_lines, train=False, data_dir=data_dir)

        return train_ds, val_ds

    @classmethod
    def calc_split_len(cls, train_pct, ds_len):
        train_rows = round(ds_len * (train_pct / 100.0))
        val_rows = ds_len - train_rows
        return train_rows, val_rows

    def __init__(self, name, data: Union[Tensor, List[str]], train, data_dir) -> None:
        """
        :param data: A list of equations strings. Each equation must have an '=' in it.
        """
        self.tokenizer = ArithmeticTokenizer(data_dir)
        self.name = name
        self.train = train
        if isinstance(data, list):
            self.data = self.tokenizer.encode(data)
        else:
            self.data = data

    def __len__(self) -> int:
        """
        :returns: total number of equations in this dataset
        """
        return self.data.shape[0]

    # @classmethod
    # def _render(cls, operand):
    #    return render(operand, join_str=" ")
    #
    # @classmethod
    # def _render_eq(parts):
    #    return " ".join(map(render, parts))


    @classmethod
    def _make_binary_operation_data(cls, operator: str, operands=None) -> List[str]:
        """
        Generate all equations "a operator b = c mod p" for each p in MODULI.
        Return a flat list of strings without EOS tokens.

        :param operator: one of "+", "-", "/"
        :param operands: (not used here, included for signature consistency)
        :returns: list of strings, each of the form "a o b = c mod p"
        """
        eqs: List[str] = []

        # Only handle the three arithmetic operators "+", "-", "/"
        if operator in ["+", "-", "/"]:
            for p in MODULI:
                # Build list of Mod(0..p-1, p)
                elems = [Mod(i, p) for i in range(p)]
                for a_mod in elems:
                    for b_mod in elems:
                        if operator == "/":
                            # Skip division by zero
                            if b_mod._value == 0:
                                continue
                            # Compute inverse of b_mod modulo p
                            inv_b = pow(b_mod._value, -1, p)
                            # c = b^{-1} * a (mod p)
                            c_val = (inv_b * a_mod._value) % p
                            c_mod = Mod(c_val, p)
                            # Append "a / b = c mod p"
                            eqs.append(f"{render(a_mod)} {operator} {render(b_mod)} = {render(c_mod)}")
                        else:
                            # Compute c for "+" or "-"
                            if operator == "+":
                                c_val = (a_mod._value + b_mod._value) % p
                            else:  # operator == "-"
                                c_val = (a_mod._value - b_mod._value) % p
                            c_mod = Mod(c_val, p)
                            # Append "a + b = c mod p" or "a - b = c mod p"
                            eqs.append(f"{render(a_mod)} {operator} {render(b_mod)} = {render(c_mod)}")
            return eqs

        # For any other operator (e.g., "s5", "*", etc.), return empty list
        return eqs

    # @staticmethod
    # def _render_unop_example(operator, lhs, rhs):
    #    return " ".join([operator, render(lhs), "=", render(rhs)])

    @staticmethod
    def _make_unary_operation_data(operator: str, operands: Tensor) -> List[str]:
        """
        :param operator: The unary operator to apply to each operand e.g. '+'
        :param operands: A tensor of operands
        :returns: list of equations"""
        num_examples = len(operands)

        if operator == "sort":
            rhs = torch.sort(operands, dim=1)[0]
        elif operator == "reverse":
            rhs = torch.flip(operands, dims=(1,))
        elif operator == "copy":
            rhs = operands
        else:
            raise Exception("unsupported operator")

        def func(L, R):
            L = map(str, L)
            R = map(str, R)
            return f"{operator} {' '.join(L)} = {' '.join(R)}"

        if num_examples < 1000000000:
            eqs = [
                func(L, R)
                for L, R in tqdm(
                    zip(operands.tolist(), rhs.tolist()), total=num_examples
                )
            ]
        else:
            with ProcessPoolExecutor() as executor:
                eqs = executor.map(func, tqdm(zip(operands, rhs), total=num_examples))

        return eqs

    # @staticmethod
    # def _make_s5_data(abstract=False) -> List[str]:
    #    elems = itertools.permutations([0, 1, 2, 3, 4])
    #    pairs = itertools.product(elems, repeat=2)
    #    eqs = []
    #    for a, b in pairs:
    #        a = np.array(a)
    #        b = np.array(b)
    #        c = b[a]
    #        eq = " ".join(map(render, (a, "s5", b, "=", c)))
    #        eq = cls._render_eq([a, , b, "=", c])
    #        eqs.append(eq)
    #
    #    return eqs

    @classmethod
    def get_dsname(cls, operator, operand_length) -> str:
        operator, noise_level = cls._get_operator_and_noise_level(operator)
        ds_name = VALID_OPERATORS[operator]
        if operand_length is not None:
            ds_name += f"_length-{operand_length}"
        if noise_level > 0:
            ds_name += f"_noise-{noise_level}"
        return ds_name

    @classmethod
    def get_file_path(cls, operator, operand_length=None, data_dir=DEFAULT_DATA_DIR):
        ds_name = cls.get_dsname(operator, operand_length)
        ds_file = bf.join(data_dir, f"{ds_name}_data.txt")
        return ds_file, ds_name

    @classmethod
    def _get_operator_and_noise_level(cls, operator):
        if "_noisy" in operator:
            operator, noise_level = operator.split("_noisy_")
            return operator, int(noise_level)
        else:
            return operator, 0



    @classmethod
    def make_data(cls, operator, operands=None, shuffle=True, seed=0) -> List[str]:
        """
        Generate all equations (wrapped with EOS_TOKEN) for the given operator.
        Then split into train/val/test (e.g., 80/10/10) and write each to separate files:
          - "<operator_name>_train.txt"
          - "<operator_name>_val.txt"
          - "<operator_name>_test.txt"
        Return the list of train equations (caller may ignore return value).
        """
        operator_key, noise_level = cls._get_operator_and_noise_level(operator)
        assert operator_key in VALID_OPERATORS

        # 1) Generate the raw list of equations (without EOS for now)
        #if operator_key not in ["sort", "reverse", "copy"]:
        if operator_key in ["+", "-", "/"]:
            raw_eqs = cls._make_binary_operation_data(operator_key)
        else:
            pass
            #raw_eqs = cls._make_unary_operation_data(operator_key, operands)

        # 2) Wrap each equation with EOS tokens
        all_eqs = [f"{eq}" for eq in raw_eqs]

        # 3) Shuffle the list (deterministically if shuffle=True and fixed seed)
        rng = np.random.RandomState(seed=seed)
        if shuffle:
            rng.shuffle(all_eqs)

        # 4) Compute split indices: 80% train, 10% val, 10% test
        total = len(all_eqs)
        n_train = int(total * 0.8)
        n_test = int(total * 0.1)
        n_val = total - n_train - n_test

        train_eqs = all_eqs[:n_train]
        val_eqs = all_eqs[n_train: n_train + n_val]
        test_eqs = all_eqs[n_train + n_val:]

        # 5) Write to files under data_dir
        #    File names: "<operator_name>_train.txt", "<operator_name>_val.txt", "<operator_name>_test.txt"
        #    Use VALID_OPERATORS 映射到的 ds_name 作为文件前缀
        ds_base = VALID_OPERATORS[operator_key]
        train_path = bf.join(cls._data_dir, f"{ds_base}_train.txt")
        val_path = bf.join(cls._data_dir, f"{ds_base}_val.txt")
        test_path = bf.join(cls._data_dir, f"{ds_base}_test.txt")

        # Ensure data directory exists (in case caller没提前创建)
        os.makedirs(cls._data_dir, exist_ok=True)

        # Write files
        with open(train_path, "w", encoding="utf-8") as f_tr:
            for eq in train_eqs:
                f_tr.write(eq + "\n")
        with open(val_path, "w", encoding="utf-8") as f_v:
            for eq in val_eqs:
                f_v.write(eq + "\n")
        with open(test_path, "w", encoding="utf-8") as f_te:
            for eq in test_eqs:
                f_te.write(eq + "\n")

        # 6) Return train set (if需要给调用者用)
        return train_eqs



    @classmethod
    def _make_lists(cls, sizes=[2, 3], nums=NUMS):
        lists: dict = {}
        for size in sizes:
            lists[size] = torch.tensor(
                list(itertools.permutations(nums, r=size)),
                dtype=torch.int,
            )
        return lists


class ArithmeticIterator(torch.utils.data.IterableDataset):
    """
    An iterator over batches of data in an ArithmeticDataset
    """

    def __init__(
        self,
        dataset: ArithmeticDataset,
        device: torch.device,
        batchsize_hint: float = 0,
        shuffle: bool = True,
    ) -> None:
        """
        :param dataset: the dataset to iterate over
        :param device: the torch device to send batches to
        :param batchsize_hint: * 0 means we use a default batchsize
                               * -1 means the entire dataset
                               * float between 0 and 1 means each batch is
                                 that fraction of the DS
                               * int > 1 means that specific batch size
        :param shuffle: whether or not to randomly shuffle the dataset
        """
        self.dataset = dataset
        self.batchsize = self.calculate_batchsize(
            len(dataset), batchsize_hint=batchsize_hint
        )
        self.device = device
        self.reset_iteration(shuffle=shuffle)

    @staticmethod
    def calculate_batchsize(ds_size: int, batchsize_hint: int = 0) -> int:
        """
        Calculates which batch size to use

        :param ds_size: the number of equations in the dataset
        :param batchsize_hint: * 0 means we use a default batchsize
                               * -1 means the entire dataset
                               * float between 0 and 1 means each batch is
                                 that fraction of the DS
                               * int > 1 means that specific batch size
        :returns: the actual batchsize to use
        """

        if batchsize_hint == -1:
            return ds_size
        elif batchsize_hint == 0:
            return min(512, math.ceil(ds_size / 2.0))
        elif (batchsize_hint > 0) and (batchsize_hint < 1):
            return math.ceil(ds_size * batchsize_hint)
        elif batchsize_hint > 1:
            return min(batchsize_hint, ds_size)
        else:
            raise ValueError("batchsize_hint must be >= -1")

    def reset_iteration(self, shuffle=True):
        self.index = 0
        if shuffle and self.dataset.train:
            self.permutation = torch.randperm(len(self.dataset))
        else:
            self.permutation = torch.arange(len(self.dataset))

    def __iter__(self):
        """
        :returns: this iterator
        """
        return self

    def __next__(self) -> Dict[str, Tensor]:
        """
        Returns one batch of data.

        :raises: StopIteration when we're out of data
        :returns: batch tensor of shape (self.batchsize, tokens_per_eq)
        """

        batch_begin = self.index * self.batchsize
        if batch_begin > len(self.dataset) - 1:
            self.reset_iteration()
            raise StopIteration
        indices = self.permutation[batch_begin : batch_begin + self.batchsize]
        text = self.dataset.data[indices, :-1]
        target = self.dataset.data[indices, 1:]
        batch = {"text": text.to(self.device), "target": target.to(self.device)}
        self.index += 1
        return batch

    def __len__(self) -> int:
        """
        :returns: the total number of batches
        """
        return math.ceil(len(self.dataset) / self.batchsize)
