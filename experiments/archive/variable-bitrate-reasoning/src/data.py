# data.py — Dataset generation and tokenization for arithmetic experiments

import random
from typing import Optional

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
OPERATORS = ["+", "-", "×"]
MISC = ["(", ")", "=", "?", " "]
DIGITS = [str(d) for d in range(10)]

VOCAB = SPECIAL_TOKENS + DIGITS + OPERATORS + MISC
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_TOKEN = {i: tok for tok, i in TOKEN_TO_ID.items()}
VOCAB_SIZE = len(VOCAB)

PAD_ID = TOKEN_TO_ID["<pad>"]
UNK_ID = TOKEN_TO_ID["<unk>"]
BOS_ID = TOKEN_TO_ID["<bos>"]
EOS_ID = TOKEN_TO_ID["<eos>"]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[int]:
    """Character-level tokenization over the defined vocabulary."""
    return [TOKEN_TO_ID.get(ch, UNK_ID) for ch in text]


def detokenize(ids) -> str:
    return "".join(ID_TO_TOKEN.get(int(i), "<unk>") for i in ids)


# ---------------------------------------------------------------------------
# Expression generator
# ---------------------------------------------------------------------------

def _gen_expr(
    n_ops: int,
    number_range: tuple[int, int],
    rng: random.Random,
) -> tuple[str, int]:
    """Recursively build an arithmetic expression with exactly n_ops operations.

    Returns (expression_string, integer_value).
    Sub-expressions with their own operators are wrapped in parentheses.
    """
    if n_ops == 0:
        n = rng.randint(number_range[0], number_range[1])
        return str(n), n

    # Distribute the n_ops-1 remaining ops between left and right subtrees.
    left_ops = rng.randint(0, n_ops - 1)
    right_ops = n_ops - 1 - left_ops

    left_str, left_val = _gen_expr(left_ops, number_range, rng)
    right_str, right_val = _gen_expr(right_ops, number_range, rng)

    op = rng.choice(["+", "-", "×"])
    if op == "+":
        val = left_val + right_val
    elif op == "-":
        val = left_val - right_val
    else:
        val = left_val * right_val

    # Parenthesise compound sub-expressions to force evaluation order.
    if left_ops > 0:
        left_str = f"({left_str})"
    if right_ops > 0:
        right_str = f"({right_str})"

    return f"{left_str} {op} {right_str}", val


def generate_arithmetic_problem(
    depth: Optional[int] = None,
    min_depth: int = 2,
    max_depth: int = 5,
    number_range: tuple[int, int] = (1, 20),
    rng: Optional[random.Random] = None,
) -> tuple[str, str, int]:
    """Return (problem_string, answer_string, depth).

    depth = number of binary operations in the expression.
    Example: ("(3 + 5) × 2 = ?", "16", 2)
    """
    if rng is None:
        rng = random.Random()
    if depth is None:
        depth = rng.randint(min_depth, max_depth)

    expr, value = _gen_expr(depth, number_range, rng)
    problem = f"{expr} = ?"
    answer = str(value)
    return problem, answer, depth


def compute_difficulty(depth: int) -> int:
    """Map operation count to difficulty label: 0=easy, 1=medium, 2=hard."""
    if depth <= 2:
        return 0
    if depth <= 4:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ArithmeticDataset(Dataset):
    def __init__(
        self,
        num_examples: int = 10_000,
        min_depth: int = 2,
        max_depth: int = 5,
        number_range: tuple[int, int] = (1, 20),
        seed: int = 42,
    ):
        rng = random.Random(seed)
        self.examples = [
            generate_arithmetic_problem(
                min_depth=min_depth,
                max_depth=max_depth,
                number_range=number_range,
                rng=rng,
            )
            for _ in range(num_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        problem, answer, depth = self.examples[idx]
        input_ids = tokenize(problem)
        target_ids = tokenize(answer)
        difficulty = compute_difficulty(depth)
        return input_ids, target_ids, difficulty


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Build padded LM sequences from a list of (input_ids, target_ids, difficulty).

    Each full sequence is:  problem_tokens + answer_tokens + [EOS]

    Returns:
        padded_ids:   (B, max_len)  — full sequences, PAD-filled
        prob_lengths: (B,)          — token count of each problem string
        difficulties: (B,)
    """
    input_ids_list, target_ids_list, difficulties = zip(*batch)

    full_seqs = [
        list(inp) + list(tgt) + [EOS_ID]
        for inp, tgt in zip(input_ids_list, target_ids_list)
    ]
    prob_lengths = [len(inp) for inp in input_ids_list]

    max_len = max(len(s) for s in full_seqs)
    padded = torch.full((len(full_seqs), max_len), PAD_ID, dtype=torch.long)
    for i, seq in enumerate(full_seqs):
        padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    return (
        padded,
        torch.tensor(prob_lengths, dtype=torch.long),
        torch.tensor(difficulties, dtype=torch.long),
    )
