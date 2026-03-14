"""Data generation for Crystal Lattice using RDKit.

Generates linear alkanes (Phase 1) and macrocyclic rings (Phase 2/3)
with 3D conformer embedding, energy filtering, and head-to-tail
distance computation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


@dataclass
class MoleculeRecord:
    """Single molecule with 3D geometry and metadata."""
    smiles: str
    atoms: list[str]
    coords_3d: np.ndarray          # (n_atoms, 3)
    head_tail_distance: float      # Angstroms
    is_ring: bool
    energy: float                  # kcal/mol (MMFF)


class MoleculeGenerator:
    """Generate and embed molecules for Crystal Lattice training."""

    def __init__(
        self,
        seed: int = 42,
        num_conformers: int = 10,
        max_iters: int = 500,
        energy_filter_sigma: float = 2.0,
    ):
        self.seed = seed
        self.num_conformers = num_conformers
        self.max_iters = max_iters
        self.energy_filter_sigma = energy_filter_sigma
        self.rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # SMILES generators
    # ------------------------------------------------------------------

    @staticmethod
    def linear_alkane_smiles(n_carbons: int) -> str:
        """Return SMILES for a straight-chain alkane with *n_carbons* carbons."""
        return "C" * n_carbons

    @staticmethod
    def macrocycle_smiles(ring_size: int) -> str:
        """Return SMILES for a simple cycloalkane of given ring size."""
        if ring_size < 3:
            raise ValueError("Ring size must be >= 3")
        # C1CCCCC...C1  with (ring_size - 1) interior C's
        return "C1" + "C" * (ring_size - 2) + "C1"

    @staticmethod
    def substituted_macrocycle_smiles(
        ring_size: int, substituent: str = "C(C)(C)C"
    ) -> str:
        """Macrocycle with a substituent (default: tert-butyl) on atom 0."""
        if ring_size < 5:
            raise ValueError("Ring too small for substituent")
        core = "C1" + "C" * (ring_size - 2) + "C1"
        # Insert substituent after the first carbon
        return "C(" + substituent + ")1" + "C" * (ring_size - 2) + "C1"

    # ------------------------------------------------------------------
    # 3D embedding
    # ------------------------------------------------------------------

    def _embed_and_optimise(
        self, mol: Chem.Mol
    ) -> list[tuple[np.ndarray, float]]:
        """Embed multiple conformers, MMFF-optimise, return (coords, energy) list."""
        mol = Chem.AddHs(mol)
        conf_ids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=self.num_conformers,
            maxAttempts=self.max_iters,
            randomSeed=self.seed,
            pruneRmsThresh=0.5,
        )
        if len(conf_ids) == 0:
            return []

        results: list[tuple[np.ndarray, float]] = []
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff_props is None:
            return []

        for cid in conf_ids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=cid)
            if ff is None:
                continue
            ff.Minimize(maxIts=self.max_iters)
            energy = ff.CalcEnergy()
            conf = mol.GetConformer(cid)
            # Only keep heavy-atom positions
            heavy_idxs = [
                i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
            ]
            coords = np.array(
                [list(conf.GetAtomPosition(i)) for i in heavy_idxs],
                dtype=np.float32,
            )
            results.append((coords, energy))
        return results

    # ------------------------------------------------------------------
    # Energy filter
    # ------------------------------------------------------------------

    def _energy_filter(
        self, conformers: list[tuple[np.ndarray, float]]
    ) -> list[tuple[np.ndarray, float]]:
        """Discard conformations with energy > mean + sigma * std."""
        if len(conformers) <= 1:
            return conformers
        energies = np.array([e for _, e in conformers])
        mu, sigma = energies.mean(), energies.std()
        if sigma < 1e-6:
            return conformers
        threshold = mu + self.energy_filter_sigma * sigma
        return [(c, e) for c, e in conformers if e <= threshold]

    # ------------------------------------------------------------------
    # Record construction
    # ------------------------------------------------------------------

    def _has_ring(self, smiles: str) -> bool:
        """Detect ring closure digits in SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return mol.GetRingInfo().NumRings() > 0

    def _head_tail_distance(self, coords: np.ndarray) -> float:
        """Euclidean distance between first and last heavy atom."""
        return float(np.linalg.norm(coords[0] - coords[-1]))

    def smiles_to_records(self, smiles: str) -> list[MoleculeRecord]:
        """Convert a SMILES string to a list of MoleculeRecords (one per valid conformer)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        conformers = self._embed_and_optimise(mol)
        conformers = self._energy_filter(conformers)
        if not conformers:
            return []

        is_ring = self._has_ring(smiles)
        records = []
        for coords, energy in conformers:
            records.append(
                MoleculeRecord(
                    smiles=smiles,
                    atoms=atom_symbols,
                    coords_3d=coords,
                    head_tail_distance=self._head_tail_distance(coords),
                    is_ring=is_ring,
                    energy=energy,
                )
            )
        return records

    # ------------------------------------------------------------------
    # Batch generators
    # ------------------------------------------------------------------

    def generate_linear_alkanes(
        self, min_carbons: int = 3, max_carbons: int = 12, count: int = 100
    ) -> list[MoleculeRecord]:
        """Generate *count* linear-alkane conformers across a range of chain lengths."""
        all_records: list[MoleculeRecord] = []
        sizes = list(range(min_carbons, max_carbons + 1))
        idx = 0
        while len(all_records) < count:
            n = sizes[idx % len(sizes)]
            smi = self.linear_alkane_smiles(n)
            recs = self.smiles_to_records(smi)
            all_records.extend(recs)
            idx += 1
            if idx > count * 5:
                break  # safety valve
        return all_records[:count]

    def generate_macrocycles(
        self,
        min_ring: int = 12,
        max_ring: int = 20,
        count: int = 50,
        with_substituent: bool = False,
    ) -> list[MoleculeRecord]:
        """Generate macrocyclic conformers."""
        all_records: list[MoleculeRecord] = []
        sizes = list(range(min_ring, max_ring + 1))
        idx = 0
        while len(all_records) < count:
            n = sizes[idx % len(sizes)]
            if with_substituent:
                smi = self.substituted_macrocycle_smiles(n)
            else:
                smi = self.macrocycle_smiles(n)
            recs = self.smiles_to_records(smi)
            all_records.extend(recs)
            idx += 1
            if idx > count * 5:
                break
        return all_records[:count]

    @staticmethod
    def validate_smiles(smiles: str) -> bool:
        """Check whether a SMILES string is chemically valid via RDKit."""
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None


def records_to_dicts(records: list[MoleculeRecord]) -> list[dict]:
    """Convert MoleculeRecords to plain dicts (for serialisation / dataloading)."""
    return [
        {
            "smiles": r.smiles,
            "atoms": r.atoms,
            "coords_3d": r.coords_3d,
            "head_tail_distance": r.head_tail_distance,
            "is_ring": r.is_ring,
            "energy": r.energy,
        }
        for r in records
    ]
