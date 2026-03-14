"""VSA Lattice encoder for Crystal Lattice.

Encodes SMILES strings into 10,000-D Holographic Reduced Representation
(HRR) hypervectors using torchhd, with a learned projection to 256-D
for downstream CLN consumption.

Key feature: the "Wormhole Operator" binds ring-closure atoms with a
CLOSURE_TAG hypervector, signalling topological adjacency to the CLN.
"""

from __future__ import annotations

import re
from typing import Optional

import torch
import torch.nn as nn
import torchhd


# Atom types we expect to encounter (extendable)
DEFAULT_ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "H"]

# Maximum sequence length (SMILES positions)
MAX_SEQ_LEN = 128


class VSALattice(nn.Module):
    """Encode SMILES into HRR hypervectors with ring-closure binding.

    Parameters
    ----------
    dim : int
        Hypervector dimensionality (default 10_000).
    proj_dim : int
        Projection target dimensionality for CLN input (default 256).
    atom_types : list[str]
        Known atom symbols.
    max_seq_len : int
        Maximum number of SMILES positions to encode.
    """

    def __init__(
        self,
        dim: int = 10_000,
        proj_dim: int = 256,
        atom_types: list[str] | None = None,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        super().__init__()
        self.dim = dim
        self.proj_dim = proj_dim
        self.atom_types = atom_types or DEFAULT_ATOM_TYPES
        self.max_seq_len = max_seq_len

        # Codebooks ----------------------------------------------------------
        # Atom-type hypervectors (one per element symbol)
        atom_hvs = torchhd.random(len(self.atom_types), dim)
        self.register_buffer("atom_hvs", atom_hvs)
        self._atom_to_idx = {s: i for i, s in enumerate(self.atom_types)}

        # Positional hypervectors (one per sequence position)
        pos_hvs = torchhd.random(max_seq_len, dim)
        self.register_buffer("pos_hvs", pos_hvs)

        # Closure-tag hypervector (unique, random)
        closure_tag = torchhd.random(1, dim)
        self.register_buffer("closure_tag", closure_tag)

        # Learned projection: 10,000-D -> proj_dim --------------------------
        self.projection = nn.Linear(dim, proj_dim)

    # ------------------------------------------------------------------
    # SMILES parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_smiles_tokens(smiles: str) -> list[dict]:
        """Parse SMILES into a list of token dicts.

        Each dict has:
            symbol : str  -- atom symbol or None for non-atom tokens
            pos    : int  -- index in the token list
            ring_ids : list[str] -- ring-closure digits attached to this atom
        """
        tokens: list[dict] = []
        i = 0
        pos = 0
        n = len(smiles)

        while i < n:
            ch = smiles[i]

            # Skip brackets, charges, chirality markers, bond chars
            if ch in ("(", ")", "[", "]", "=", "#", "\\", "/", "+", "-", "@", "."):
                i += 1
                continue

            # Two-letter elements
            if ch.isupper() and i + 1 < n and smiles[i + 1].islower() and smiles[i + 1] not in "0123456789":
                symbol = smiles[i : i + 2]
                i += 2
            elif ch.isupper():
                symbol = ch
                i += 1
            else:
                i += 1
                continue

            # Collect trailing ring-closure digits (including %nn notation)
            ring_ids: list[str] = []
            while i < n:
                if smiles[i].isdigit():
                    ring_ids.append(smiles[i])
                    i += 1
                elif smiles[i] == "%" and i + 2 < n and smiles[i + 1].isdigit() and smiles[i + 2].isdigit():
                    ring_ids.append(smiles[i + 1 : i + 3])
                    i += 3
                else:
                    break

            tokens.append({"symbol": symbol, "pos": pos, "ring_ids": ring_ids})
            pos += 1

        return tokens

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _atom_hv(self, symbol: str) -> torch.Tensor:
        """Look up (or generate) the hypervector for an atom symbol."""
        idx = self._atom_to_idx.get(symbol)
        if idx is not None:
            return self.atom_hvs[idx]
        # Unknown atom: hash-based fallback (deterministic)
        gen = torch.Generator()
        gen.manual_seed(hash(symbol) % (2**31))
        return torch.randn(self.dim, generator=gen, device=self.atom_hvs.device)

    def encode_smiles(self, smiles: str) -> torch.Tensor:
        """Encode a single SMILES string into a 10,000-D HRR hypervector.

        Steps:
            1. Parse SMILES to atom tokens with positions and ring IDs.
            2. For each atom: bind atom_hv with positional_hv and bundle into mol_hv.
            3. For ring closures: bind positions of both closure atoms with CLOSURE_TAG.
        """
        tokens = self.parse_smiles_tokens(smiles)
        if not tokens:
            return torch.zeros(self.dim, device=self.atom_hvs.device)

        mol_hv = torch.zeros(self.dim, device=self.atom_hvs.device)

        # Map ring_id -> list of positions for closure detection
        ring_map: dict[str, list[int]] = {}
        for tok in tokens:
            for rid in tok["ring_ids"]:
                ring_map.setdefault(rid, []).append(tok["pos"])

        # Bundle atom-position bindings
        for tok in tokens:
            pos = min(tok["pos"], self.max_seq_len - 1)
            a_hv = self._atom_hv(tok["symbol"])
            p_hv = self.pos_hvs[pos]
            # HRR binding = circular convolution (element-wise multiply in frequency domain)
            bound = torchhd.bind(a_hv.unsqueeze(0), p_hv.unsqueeze(0)).squeeze(0)
            mol_hv = torchhd.bundle(mol_hv.unsqueeze(0), bound.unsqueeze(0)).squeeze(0)

        # Wormhole operator: ring closure binding
        ct = self.closure_tag.squeeze(0)
        for rid, positions in ring_map.items():
            if len(positions) >= 2:
                p_i = min(positions[0], self.max_seq_len - 1)
                p_j = min(positions[-1], self.max_seq_len - 1)
                # Bind the two positional HVs with the closure tag
                closure_hv = torchhd.bind(
                    torchhd.bind(
                        self.pos_hvs[p_i].unsqueeze(0),
                        self.pos_hvs[p_j].unsqueeze(0),
                    ),
                    ct.unsqueeze(0),
                ).squeeze(0)
                mol_hv = torchhd.bundle(mol_hv.unsqueeze(0), closure_hv.unsqueeze(0)).squeeze(0)

        return mol_hv

    def encode_and_project(self, smiles: str) -> torch.Tensor:
        """Encode SMILES and project to CLN dimensionality (256-D)."""
        hv = self.encode_smiles(smiles)
        return self.projection(hv)

    def encode_batch(self, smiles_list: list[str]) -> torch.Tensor:
        """Encode a batch of SMILES, returning (B, dim) raw hypervectors."""
        hvs = torch.stack([self.encode_smiles(s) for s in smiles_list])
        return hvs

    def encode_and_project_batch(self, smiles_list: list[str]) -> torch.Tensor:
        """Encode and project a batch, returning (B, proj_dim)."""
        hvs = self.encode_batch(smiles_list)
        return self.projection(hvs)

    # ------------------------------------------------------------------
    # Integrity / angle concentration
    # ------------------------------------------------------------------

    @staticmethod
    def angle_concentration(hvs: torch.Tensor) -> float:
        """Compute angle concentration: mean pairwise cosine similarity.

        This measures how "crystallised" a set of hypervectors is.
        High concentration = low diversity (vectors pointing same way).
        Low concentration = high diversity.

        Parameters
        ----------
        hvs : Tensor of shape (N, D)

        Returns
        -------
        float in [-1, 1]
        """
        if hvs.shape[0] < 2:
            return 1.0
        hvs_norm = torch.nn.functional.normalize(hvs, dim=-1)
        cos_sim = hvs_norm @ hvs_norm.T
        # Mask diagonal
        n = cos_sim.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=cos_sim.device)
        return float(cos_sim[mask].mean().item())

    def structural_integrity(
        self, hidden: torch.Tensor, anchor: torch.Tensor
    ) -> float:
        """Compute structural integrity between a latent state and the VSA anchor.

        Uses angle concentration on the pair (hidden, anchor), which for
        two vectors reduces to their cosine similarity.
        """
        pair = torch.stack(
            [
                torch.nn.functional.normalize(hidden.flatten(), dim=0),
                torch.nn.functional.normalize(anchor.flatten(), dim=0),
            ]
        )
        return self.angle_concentration(pair)
