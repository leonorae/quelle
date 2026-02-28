# data.py — Dataset generation and tokenization for arithmetic experiments
#
# Stub — implement according to the spec in the top-level README.md.

import random
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
OPERATORS = ["+", "-", "×"]
MISC = ["(", ")", "=", "?", " "]
DIGITS = [str(d) for d in range(10)]

VOCAB = SPECIAL_TOKENS + DIGITS + OPERATORS + MISC
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_TOKEN = {i: tok for tok, i in TOKEN_TO_ID.items()}
VOCAB_SIZE = len(VOCAB)


def tokenize(text: str) -> list[int]:
    """Character-level tokenization over the defined vocabulary."""
    return [TOKEN_TO_ID.get(ch, TOKEN_TO_ID["<unk>"]) for ch in text]


def detokenize(ids: list[int]) -> str:
    return "".join(ID_TO_TOKEN.get(i, "<unk>") for i in ids)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_arithmetic_problem(
    depth: int | None = None,
    min_depth: int = 2,
    max_depth: int = 5,
    number_range: tuple[int, int] = (1, 20),
    rng: random.Random | None = None,
) -> tuple[str, str, int]:
    """Return (problem_string, answer_string, depth).

    Example: ("((3 + 5) × 2) + 7 = ?", "23", 3)
    """
    # TODO: implement recursive expression generator
    raise NotImplementedError("Implement generate_arithmetic_problem.")


def compute_difficulty(depth: int) -> int:
    """Map depth to difficulty label: 0=easy, 1=medium, 2=hard."""
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
