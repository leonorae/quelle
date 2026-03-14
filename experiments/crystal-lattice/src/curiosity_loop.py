"""Curiosity Loop: integration and training script for Crystal Lattice.

Orchestrates the three-phase training process:
  Phase 1: Seed Crystallization -- train HRM on linear alkanes
  Phase 2: Curiosity Loop -- LLM-mutated stepping stones with
           diversity-entropy filtering and RDKit ground truth
  Phase 3: Super-Resolution -- test on held-out macrocycles

CLI interface with argparse. Checkpointing and metric logging.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data_generator import MoleculeGenerator, MoleculeRecord
from .vsa_lattice import VSALattice
from .resonator import ContinuousLatentNavigator, HRM, CLNDiagnostics

logger = logging.getLogger("crystal-lattice")


# ======================================================================
# Dataset
# ======================================================================

class MoleculeDataset(Dataset):
    """Wraps a list of MoleculeRecords for DataLoader consumption."""

    def __init__(self, records: list[MoleculeRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        return {
            "smiles": r.smiles,
            "head_tail_distance": r.head_tail_distance,
            "is_ring": float(r.is_ring),
        }


def collate_molecules(batch: list[dict]) -> dict:
    """Custom collation: keep SMILES as list, stack scalars."""
    return {
        "smiles": [b["smiles"] for b in batch],
        "head_tail_distance": torch.tensor(
            [b["head_tail_distance"] for b in batch], dtype=torch.float32
        ),
        "is_ring": torch.tensor(
            [b["is_ring"] for b in batch], dtype=torch.float32
        ),
    }


# ======================================================================
# Ollama interface for SMILES mutation
# ======================================================================

def query_ollama(
    prompt: str,
    model: str = "llama3.2:1b",
    host: str = "http://localhost:11434",
    timeout: int = 60,
) -> str:
    """Query a local Ollama instance and return the response text."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.9, "num_predict": 512},
    })
    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", f"{host}/api/generate",
             "-H", "Content-Type: application/json",
             "-d", payload],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            logger.warning(f"Ollama curl failed: {result.stderr}")
            return ""
        resp = json.loads(result.stdout)
        return resp.get("response", "")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"Ollama query failed: {e}")
        return ""


def generate_mutations(
    seed_smiles: list[str],
    num_mutations: int = 50,
    model: str = "llama3.2:1b",
    host: str = "http://localhost:11434",
) -> list[str]:
    """Use Ollama to generate macrocyclic SMILES mutations from seed linear chains.

    Prompts the LLM to mutate linear alkane SMILES into macrocyclic rings
    of various sizes (12-20 atoms). Filters results through RDKit validity.
    """
    seeds_str = ", ".join(seed_smiles[:5])
    prompt = (
        f"You are a chemistry assistant. Given these linear alkane SMILES: {seeds_str}\n"
        f"Generate {num_mutations} different macrocyclic ring SMILES strings with ring sizes "
        f"between 12 and 20 atoms. Include some with substituents like methyl or tert-butyl groups.\n"
        f"Output ONLY the SMILES strings, one per line, with no other text or explanation.\n"
        f"Examples of valid macrocyclic SMILES: C1CCCCCCCCCCC1, C1CCCCCCCCCCCC1\n"
    )

    response = query_ollama(prompt, model=model, host=host)
    if not response:
        logger.warning("Ollama returned empty response, using programmatic fallback")
        return _programmatic_mutations(num_mutations)

    # Parse response: extract lines that look like SMILES
    candidates: list[str] = []
    for line in response.strip().split("\n"):
        line = line.strip().strip("- ").strip("`").strip()
        if not line or len(line) < 3:
            continue
        # Remove numbering prefixes like "1. " or "1) "
        if line[0].isdigit():
            for sep in [". ", ") ", ": ", " "]:
                if sep in line:
                    line = line.split(sep, 1)[1].strip()
                    break
        # RDKit validity check (from REVIEW.md: add validity before entropy scoring)
        if MoleculeGenerator.validate_smiles(line):
            candidates.append(line)

    if len(candidates) < 10:
        logger.warning(
            f"Only {len(candidates)} valid SMILES from LLM, supplementing with programmatic mutations"
        )
        candidates.extend(_programmatic_mutations(num_mutations - len(candidates)))

    return candidates[:num_mutations]


def _programmatic_mutations(count: int) -> list[str]:
    """Fallback: generate macrocyclic SMILES programmatically."""
    gen = MoleculeGenerator()
    mutations: list[str] = []
    for ring_size in range(12, 21):
        mutations.append(gen.macrocycle_smiles(ring_size))
        if ring_size >= 14:
            mutations.append(gen.substituted_macrocycle_smiles(ring_size))
            mutations.append(gen.substituted_macrocycle_smiles(ring_size, "C"))
            mutations.append(gen.substituted_macrocycle_smiles(ring_size, "CC"))
    # Deduplicate and trim
    mutations = list(dict.fromkeys(mutations))
    return mutations[:count]


# ======================================================================
# Diversity-entropy scoring
# ======================================================================

def diversity_entropy_score(
    hrm: HRM,
    candidate_smiles: list[str],
    training_smiles: list[str],
    device: torch.device,
) -> list[tuple[str, float]]:
    """Score candidates by diversity-entropy: Score = entropy * (1 - max_cosine_sim_to_training).

    Parameters
    ----------
    hrm : the trained HRM model
    candidate_smiles : SMILES to score
    training_smiles : existing training set SMILES
    device : torch device

    Returns
    -------
    List of (smiles, score) sorted descending by score.
    """
    if not candidate_smiles:
        return []

    hrm.eval()
    with torch.no_grad():
        # Encode candidates
        cand_hvs = hrm.vsa.encode_batch(candidate_smiles).to(device)  # (C, D)
        # Encode training set
        if training_smiles:
            train_hvs = hrm.vsa.encode_batch(training_smiles).to(device)  # (T, D)
        else:
            train_hvs = torch.zeros(1, hrm.vsa.dim, device=device)

        # Normalise for cosine similarity
        cand_norm = F.normalize(cand_hvs, dim=-1)
        train_norm = F.normalize(train_hvs, dim=-1)

        # Max cosine similarity of each candidate to training set
        cos_sim = cand_norm @ train_norm.T  # (C, T)
        max_sim = cos_sim.max(dim=1).values  # (C,)

        # Get entropy from CLN forward pass
        projected = hrm.vsa.projection(cand_hvs)  # (C, proj_dim)
        _, _, diagnostics = hrm.cln(projected)
        # Use final-iteration entropy
        final_entropy = diagnostics[-1].entropy if diagnostics else 0.0

        # Per-candidate entropy: run individually for accurate per-sample entropy
        scores: list[tuple[str, float]] = []
        for i, smi in enumerate(candidate_smiles):
            proj_i = projected[i : i + 1]  # (1, proj_dim)
            _, _, diag_i = hrm.cln(proj_i)
            ent_i = diag_i[-1].entropy if diag_i else 0.0
            sim_i = float(max_sim[i].item())
            score = ent_i * (1.0 - sim_i)
            scores.append((smi, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ======================================================================
# Training utilities
# ======================================================================

def train_epoch(
    hrm: HRM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ring_loss_weight: float = 1.0,
) -> dict:
    """Train HRM for one epoch. Returns dict of average losses."""
    hrm.train()
    total_dist_loss = 0.0
    total_ring_loss = 0.0
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        smiles = batch["smiles"]
        target_dist = batch["head_tail_distance"].to(device)   # (B,)
        target_ring = batch["is_ring"].to(device)              # (B,)

        optimizer.zero_grad()

        pred_dist, ring_logit, _ = hrm(smiles)
        pred_dist = pred_dist.squeeze(-1)         # (B,)
        ring_logit = ring_logit.squeeze(-1)       # (B,)

        dist_loss = F.mse_loss(pred_dist, target_dist)
        ring_loss = F.binary_cross_entropy_with_logits(ring_logit, target_ring)
        loss = dist_loss + ring_loss_weight * ring_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(hrm.parameters(), max_norm=5.0)
        optimizer.step()

        total_dist_loss += dist_loss.item()
        total_ring_loss += ring_loss.item()
        total_loss += loss.item()
        n_batches += 1

    if n_batches == 0:
        return {"dist_loss": 0.0, "ring_loss": 0.0, "total_loss": 0.0}

    return {
        "dist_loss": total_dist_loss / n_batches,
        "ring_loss": total_ring_loss / n_batches,
        "total_loss": total_loss / n_batches,
    }


def evaluate(
    hrm: HRM,
    records: list[MoleculeRecord],
    device: torch.device,
) -> dict:
    """Evaluate HRM on a set of records. Returns metrics dict."""
    hrm.eval()
    if not records:
        return {"mae": 0.0, "ring_accuracy": 0.0, "n": 0}

    smiles = [r.smiles for r in records]
    true_dist = np.array([r.head_tail_distance for r in records])
    true_ring = np.array([float(r.is_ring) for r in records])

    with torch.no_grad():
        pred_dist, ring_logit, diagnostics = hrm(smiles)
        pred_dist_np = pred_dist.squeeze(-1).cpu().numpy()
        ring_prob = torch.sigmoid(ring_logit).squeeze(-1).cpu().numpy()

    mae = float(np.mean(np.abs(pred_dist_np - true_dist)))
    ring_pred = (ring_prob > 0.5).astype(float)
    ring_acc = float(np.mean(ring_pred == true_ring))

    return {
        "mae": mae,
        "ring_accuracy": ring_acc,
        "n": len(records),
        "final_entropy": diagnostics[-1].entropy if diagnostics else 0.0,
        "final_integrity": diagnostics[-1].integrity if diagnostics else 0.0,
    }


# ======================================================================
# Checkpointing
# ======================================================================

def save_checkpoint(
    hrm: HRM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
) -> None:
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "hrm_state_dict": hrm.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(
    hrm: HRM,
    optimizer: torch.optim.Optimizer,
    path: Path,
) -> int:
    """Load checkpoint, return the epoch number."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hrm.load_state_dict(ckpt["hrm_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(f"Checkpoint loaded: {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


# ======================================================================
# CuriosityLoop: main orchestrator
# ======================================================================

class CuriosityLoop:
    """Three-phase training orchestrator for Crystal Lattice."""

    def __init__(
        self,
        # Architecture
        vsa_dim: int = 10_000,
        proj_dim: int = 256,
        cln_nhead: int = 8,
        cln_ff_dim: int = 512,
        cln_layers: int = 2,
        cln_iters: int = 8,
        lora_rank: int = 8,
        # Data
        num_alkanes: int = 100,
        min_carbons: int = 3,
        max_carbons: int = 12,
        # Training
        lr: float = 1e-3,
        batch_size: int = 16,
        phase1_epochs: int = 50,
        phase2_rounds: int = 3,
        phase2_finetune_epochs: int = 20,
        ring_loss_weight: float = 1.0,
        # Curiosity
        num_mutations: int = 50,
        top_k: int = 10,
        ollama_model: str = "llama3.2:1b",
        ollama_host: str = "http://localhost:11434",
        # I/O
        output_dir: str = "outputs",
        seed: int = 42,
        device: str = "auto",
    ):
        self.config = {k: v for k, v in locals().items() if k != "self"}

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data generator
        self.data_gen = MoleculeGenerator(seed=seed)

        # Model
        self.vsa = VSALattice(dim=vsa_dim, proj_dim=proj_dim).to(self.device)
        self.cln = ContinuousLatentNavigator(
            d_model=proj_dim,
            nhead=cln_nhead,
            dim_ff=cln_ff_dim,
            num_layers=cln_layers,
            num_iters=cln_iters,
            lora_rank=lora_rank,
        ).to(self.device)
        self.hrm = HRM(self.vsa, self.cln).to(self.device)

        self.optimizer = torch.optim.AdamW(self.hrm.parameters(), lr=lr)
        self.ring_loss_weight = ring_loss_weight

        # Config references
        self.num_alkanes = num_alkanes
        self.min_carbons = min_carbons
        self.max_carbons = max_carbons
        self.batch_size = batch_size
        self.phase1_epochs = phase1_epochs
        self.phase2_rounds = phase2_rounds
        self.phase2_finetune_epochs = phase2_finetune_epochs
        self.num_mutations = num_mutations
        self.top_k = top_k
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host

        # Metric history
        self.metrics_log: list[dict] = []

    def _log_metrics(self, phase: str, epoch: int, metrics: dict) -> None:
        """Log and store metrics."""
        entry = {"phase": phase, "epoch": epoch, **metrics, "timestamp": time.time()}
        self.metrics_log.append(entry)
        logger.info(f"[{phase}] epoch={epoch} {metrics}")

    # ------------------------------------------------------------------
    # Phase 1: Seed Crystallization
    # ------------------------------------------------------------------

    def phase1(self) -> list[MoleculeRecord]:
        """Train on linear alkanes to learn basic distance prediction."""
        logger.info("=== PHASE 1: Seed Crystallization ===")

        records = self.data_gen.generate_linear_alkanes(
            min_carbons=self.min_carbons,
            max_carbons=self.max_carbons,
            count=self.num_alkanes,
        )
        logger.info(f"Generated {len(records)} linear alkane conformers")

        dataset = MoleculeDataset(records)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_molecules,
        )

        for epoch in range(self.phase1_epochs):
            losses = train_epoch(
                self.hrm, dataloader, self.optimizer, self.device,
                ring_loss_weight=self.ring_loss_weight,
            )
            if (epoch + 1) % 10 == 0 or epoch == 0:
                eval_metrics = evaluate(self.hrm, records, self.device)
                self._log_metrics("phase1", epoch + 1, {**losses, **eval_metrics})

        # Checkpoint
        save_checkpoint(
            self.hrm, self.optimizer, self.phase1_epochs,
            evaluate(self.hrm, records, self.device),
            self.output_dir / "checkpoint_phase1.pt",
        )
        return records

    # ------------------------------------------------------------------
    # Phase 2: Curiosity Loop
    # ------------------------------------------------------------------

    def phase2(self, phase1_records: list[MoleculeRecord]) -> list[MoleculeRecord]:
        """Curiosity-driven stepping stone generation and fine-tuning."""
        logger.info("=== PHASE 2: Curiosity Loop ===")

        training_smiles = list({r.smiles for r in phase1_records})
        all_stepping_stones: list[MoleculeRecord] = []

        for round_idx in range(self.phase2_rounds):
            logger.info(f"--- Curiosity round {round_idx + 1}/{self.phase2_rounds} ---")

            # Step 1: Generate mutations via Ollama (with RDKit validity check)
            seed_smiles = training_smiles[:5]
            candidates = generate_mutations(
                seed_smiles,
                num_mutations=self.num_mutations,
                model=self.ollama_model,
                host=self.ollama_host,
            )
            logger.info(f"Generated {len(candidates)} valid candidate mutations")

            if not candidates:
                logger.warning("No valid candidates generated, skipping round")
                continue

            # Step 2: Score by diversity-entropy
            scored = diversity_entropy_score(
                self.hrm, candidates, training_smiles, self.device
            )

            # Step 3: Select top-k
            top_candidates = scored[: self.top_k]
            logger.info(
                f"Top-{self.top_k} scores: "
                + ", ".join(f"{s:.3f}" for _, s in top_candidates)
            )

            # Step 4: Get RDKit ground truth for top candidates
            stepping_stones: list[MoleculeRecord] = []
            for smi, score in top_candidates:
                recs = self.data_gen.smiles_to_records(smi)
                stepping_stones.extend(recs)

            if not stepping_stones:
                logger.warning("No valid 3D conformers from top candidates")
                continue

            logger.info(f"Got {len(stepping_stones)} stepping stone conformers")
            all_stepping_stones.extend(stepping_stones)

            # Step 5: Fine-tune HRM on stepping stones
            combined = phase1_records + all_stepping_stones
            dataset = MoleculeDataset(combined)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_molecules,
            )

            for epoch in range(self.phase2_finetune_epochs):
                losses = train_epoch(
                    self.hrm, dataloader, self.optimizer, self.device,
                    ring_loss_weight=self.ring_loss_weight,
                )
                if (epoch + 1) % 5 == 0:
                    eval_metrics = evaluate(self.hrm, combined, self.device)
                    self._log_metrics(
                        f"phase2_r{round_idx + 1}", epoch + 1,
                        {**losses, **eval_metrics}
                    )

            # Update training SMILES for next round's diversity calculation
            training_smiles.extend([r.smiles for r in stepping_stones])
            training_smiles = list(set(training_smiles))

        # Checkpoint
        save_checkpoint(
            self.hrm, self.optimizer, -1,
            evaluate(self.hrm, phase1_records + all_stepping_stones, self.device),
            self.output_dir / "checkpoint_phase2.pt",
        )
        return all_stepping_stones

    # ------------------------------------------------------------------
    # Phase 3: Super-Resolution Testing
    # ------------------------------------------------------------------

    def phase3(self) -> dict:
        """Test on held-out macrocycles, including substituted rings."""
        logger.info("=== PHASE 3: Super-Resolution Testing ===")

        # Simple macrocycles
        simple_rings = self.data_gen.generate_macrocycles(
            min_ring=12, max_ring=20, count=30, with_substituent=False
        )
        # Substituted macrocycles (harder -- tert-butyl steric clash)
        sub_rings = self.data_gen.generate_macrocycles(
            min_ring=14, max_ring=20, count=20, with_substituent=True
        )

        simple_metrics = evaluate(self.hrm, simple_rings, self.device)
        sub_metrics = evaluate(self.hrm, sub_rings, self.device)

        logger.info(f"Simple macrocycles: {simple_metrics}")
        logger.info(f"Substituted macrocycles: {sub_metrics}")

        self._log_metrics("phase3_simple", 0, simple_metrics)
        self._log_metrics("phase3_substituted", 0, sub_metrics)

        results = {
            "simple_macrocycles": simple_metrics,
            "substituted_macrocycles": sub_metrics,
        }

        # Save final results
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_path}")

        # Save full metrics log
        log_path = self.output_dir / "metrics_log.json"
        with open(log_path, "w") as f:
            json.dump(self.metrics_log, f, indent=2, default=str)

        return results

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute all three phases sequentially."""
        logger.info("Starting Crystal Lattice experiment")
        logger.info(f"Device: {self.device}")
        logger.info(f"Config: {json.dumps(self.config, indent=2, default=str)}")

        # Save config
        config_path = self.output_dir / "run_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

        phase1_records = self.phase1()
        stepping_stones = self.phase2(phase1_records)
        results = self.phase3()

        # Final checkpoint
        save_checkpoint(
            self.hrm, self.optimizer, -1,
            results,
            self.output_dir / "checkpoint_final.pt",
        )

        logger.info("=== Experiment complete ===")
        return results


# ======================================================================
# CLI
# ======================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Crystal Lattice: Curiosity-Driven Resonator experiment"
    )

    # Architecture
    arch = p.add_argument_group("Architecture")
    arch.add_argument("--vsa-dim", type=int, default=10_000)
    arch.add_argument("--proj-dim", type=int, default=256)
    arch.add_argument("--cln-nhead", type=int, default=8)
    arch.add_argument("--cln-ff-dim", type=int, default=512)
    arch.add_argument("--cln-layers", type=int, default=2)
    arch.add_argument("--cln-iters", type=int, default=8)
    arch.add_argument("--lora-rank", type=int, default=8)

    # Data
    data = p.add_argument_group("Data")
    data.add_argument("--num-alkanes", type=int, default=100)
    data.add_argument("--min-carbons", type=int, default=3)
    data.add_argument("--max-carbons", type=int, default=12)

    # Training
    train = p.add_argument_group("Training")
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--phase1-epochs", type=int, default=50)
    train.add_argument("--phase2-rounds", type=int, default=3)
    train.add_argument("--phase2-finetune-epochs", type=int, default=20)
    train.add_argument("--ring-loss-weight", type=float, default=1.0)

    # Curiosity
    cur = p.add_argument_group("Curiosity")
    cur.add_argument("--num-mutations", type=int, default=50)
    cur.add_argument("--top-k", type=int, default=10)
    cur.add_argument("--ollama-model", type=str, default="llama3.2:1b")
    cur.add_argument("--ollama-host", type=str, default="http://localhost:11434")

    # I/O
    io = p.add_argument_group("I/O")
    io.add_argument("--output-dir", type=str, default="outputs")
    io.add_argument("--seed", type=int, default=42)
    io.add_argument("--device", type=str, default="auto")
    io.add_argument("--log-level", type=str, default="INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Phase selection
    p.add_argument("--phase", type=str, default="all",
                    choices=["all", "1", "2", "3"],
                    help="Run specific phase (default: all)")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint to resume from")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(args.output_dir) / "experiment.log", mode="a"),
        ],
    )

    loop = CuriosityLoop(
        vsa_dim=args.vsa_dim,
        proj_dim=args.proj_dim,
        cln_nhead=args.cln_nhead,
        cln_ff_dim=args.cln_ff_dim,
        cln_layers=args.cln_layers,
        cln_iters=args.cln_iters,
        lora_rank=args.lora_rank,
        num_alkanes=args.num_alkanes,
        min_carbons=args.min_carbons,
        max_carbons=args.max_carbons,
        lr=args.lr,
        batch_size=args.batch_size,
        phase1_epochs=args.phase1_epochs,
        phase2_rounds=args.phase2_rounds,
        phase2_finetune_epochs=args.phase2_finetune_epochs,
        ring_loss_weight=args.ring_loss_weight,
        num_mutations=args.num_mutations,
        top_k=args.top_k,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )

    # Load checkpoint if specified
    if args.checkpoint:
        load_checkpoint(loop.hrm, loop.optimizer, Path(args.checkpoint))

    if args.phase == "all":
        loop.run()
    elif args.phase == "1":
        loop.phase1()
    elif args.phase == "2":
        # Phase 2 needs phase 1 data
        records = loop.data_gen.generate_linear_alkanes(
            min_carbons=loop.min_carbons,
            max_carbons=loop.max_carbons,
            count=loop.num_alkanes,
        )
        loop.phase2(records)
    elif args.phase == "3":
        loop.phase3()


if __name__ == "__main__":
    main()
