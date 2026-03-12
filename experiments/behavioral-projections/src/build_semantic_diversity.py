"""Build semantic diversity corpus component (D8 component 4, D11 visual grounding).

Generates a large candidate pool of prompts across semantic axes, embeds them
with a sentence transformer, clusters via k-means, and samples uniformly from
clusters to maximize diversity.

    python -m src.build_semantic_diversity [--output-dir prompts] [--target 2000] [--seed 42]

Produces: semantic_diversity_full.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans


# --- Candidate pool generation ---
# We define semantic axes and generate prompts by combining topics with templates.
# This is intentionally broad — the clustering step handles deduplication.

TEMPLATES_FACTUAL = [
    "What is {}?",
    "Explain {} in detail.",
    "How does {} work?",
    "What are the key properties of {}?",
    "Compare and contrast {} with its closest alternatives.",
]

TEMPLATES_CREATIVE = [
    "Write a short story involving {}.",
    "Describe {} from the perspective of someone encountering it for the first time.",
    "What metaphor best captures the essence of {}?",
    "How would you teach a child about {}?",
    "What are common misconceptions about {}?",
]

TEMPLATES_REASONING = [
    "What would happen if {} suddenly disappeared?",
    "What is the strongest argument against {}?",
    "How has {} changed over the last century?",
    "What is the relationship between {} and human wellbeing?",
    "If you could redesign {}, what would you change and why?",
]

TEMPLATES_VISUAL = [
    "Describe what {} looks like in vivid detail.",
    "What colors and shapes are associated with {}?",
    "Paint a mental picture of {} at sunset.",
    "How would you photograph {} to convey its essence?",
    "Describe the visual difference between {} and something commonly confused with it.",
]

ALL_TEMPLATES = TEMPLATES_FACTUAL + TEMPLATES_CREATIVE + TEMPLATES_REASONING + TEMPLATES_VISUAL

# Topics organized by domain. Visual-grounding cluster is explicit per D11.
TOPICS = {
    "mathematics": [
        "prime numbers", "topology", "Bayesian inference", "fractals",
        "eigenvalues", "the Riemann hypothesis", "graph theory",
        "differential equations", "combinatorics", "measure theory",
        "the central limit theorem", "group symmetry", "optimization",
        "stochastic processes", "number theory", "linear algebra",
        "Gödel's incompleteness theorems", "the Monty Hall problem",
        "non-Euclidean geometry", "chaos theory", "Fourier transforms",
    ],
    "physics": [
        "quantum entanglement", "general relativity", "entropy",
        "superconductivity", "the Higgs boson", "dark matter",
        "wave-particle duality", "the arrow of time", "phase transitions",
        "turbulence", "the Casimir effect", "cosmic inflation",
    ],
    "biology": [
        "photosynthesis", "CRISPR gene editing", "mitochondria",
        "neural plasticity", "evolution by natural selection",
        "the microbiome", "cell division", "protein folding",
        "epigenetics", "symbiosis", "circadian rhythms",
    ],
    "computer_science": [
        "neural networks", "cryptography", "distributed systems",
        "compiler design", "the halting problem", "garbage collection",
        "reinforcement learning", "type theory", "database indexing",
        "consensus algorithms", "attention mechanisms", "cache coherence",
    ],
    "philosophy": [
        "consciousness", "free will", "the trolley problem",
        "existentialism", "the ship of Theseus", "epistemology",
        "moral relativism", "the hard problem of consciousness",
        "social contract theory", "phenomenology", "nihilism",
        "the is-ought gap",
    ],
    "social_science": [
        "cognitive biases", "supply and demand", "social stratification",
        "the tragedy of the commons", "game theory in economics",
        "cultural relativism", "institutional trust", "voting systems",
        "behavioral economics", "urbanization", "income inequality",
    ],
    "arts_literature": [
        "stream of consciousness writing", "impressionism",
        "the hero's journey", "jazz improvisation", "magical realism",
        "the uncanny valley", "chiaroscuro", "narrative unreliability",
        "minimalism in design", "the sublime in art", "syncopation",
    ],
    "technology": [
        "large language models", "blockchain", "quantum computing",
        "autonomous vehicles", "nuclear fusion reactors",
        "brain-computer interfaces", "3D printing", "edge computing",
        "generative adversarial networks", "the internet of things",
        "solid-state batteries",
    ],
    "everyday_concrete": [
        "a bicycle", "a kitchen knife", "a thunderstorm",
        "a cup of coffee", "a bridge", "a candle flame",
        "a wristwatch", "a garden", "a bookshelf", "a staircase",
        "a key", "a mirror", "a pencil", "a window",
        "a doorbell", "an elevator", "a campfire", "a compass",
        "a sewing needle", "a telescope", "a violin", "a lighthouse",
    ],
    # D11: visual-grounding cluster for slicer compatibility
    "visual_grounding": [
        "a coral reef", "a mountain range at dawn", "a crowded marketplace",
        "an ancient temple", "a lightning strike", "a field of sunflowers",
        "a glacier calving into the sea", "a dense fog over a lake",
        "a neon-lit city street at night", "an erupting volcano",
        "a spider web covered in dew", "the northern lights",
        "a crumbling stone wall", "a bioluminescent bay",
        "a desert sand dune", "a waterfall in a rainforest",
        "a snowy village", "an abandoned factory", "a tidal pool",
        "a cathedral interior", "a butterfly wing under magnification",
        "a satellite image of a hurricane", "rust patterns on metal",
        "ice crystals forming on a window", "a chalk cliff coastline",
        "light filtering through stained glass", "a cave with stalactites",
        "a wildfire at the tree line", "a frozen lake surface",
        "a flock of starlings in murmuration",
    ],
    "abstract_concepts": [
        "justice", "nostalgia", "emergence", "ambiguity", "resilience",
        "irony", "causality", "symmetry", "freedom", "trust",
        "complexity", "authenticity", "paradox", "intuition", "entropy",
    ],
    "processes": [
        "fermentation", "erosion", "urbanization", "debugging code",
        "learning to ride a bicycle", "negotiating a contract",
        "composing music", "diagnosing a disease", "building a campfire",
        "navigating by stars", "translating between languages",
        "crystallization", "photographic development", "distillation",
        "bread baking", "software deployment", "scientific peer review",
        "archaeological excavation", "seed germination",
    ],
    "code_and_systems": [
        "a race condition in concurrent code", "TCP/IP handshake",
        "a memory leak", "a deadlock", "API rate limiting",
        "a stack overflow", "a segmentation fault",
        "dependency injection", "eventual consistency",
        "a SQL join operation", "binary search",
        "a hash collision", "a buffer overflow",
        "load balancing", "a regular expression engine",
    ],
    "history_and_culture": [
        "the printing press", "the French Revolution",
        "the Silk Road", "the Industrial Revolution",
        "the Library of Alexandria", "the space race",
        "the Columbian exchange", "the Renaissance",
        "the fall of the Roman Empire", "the invention of writing",
        "the abolition of slavery", "the Green Revolution",
        "the Enlightenment", "oral tradition",
    ],
}


def generate_candidate_pool(seed: int) -> list[dict]:
    """Generate candidate prompts from topic × template combinations."""
    rng = random.Random(seed)
    candidates = []
    idx = 0

    for domain, topics in TOPICS.items():
        for topic in topics:
            for tmpl in ALL_TEMPLATES:
                candidates.append({
                    "prompt_id": f"sem_cand_{idx:05d}",
                    "text": tmpl.format(topic),
                    "category": "semantic",
                    "group_id": None,
                    "domain": domain,
                })
                idx += 1

    rng.shuffle(candidates)
    return candidates


def embed_and_cluster(
    candidates: list[dict],
    n_clusters: int,
    samples_per_cluster: int,
    seed: int,
) -> list[dict]:
    """Embed candidates, cluster, and sample uniformly from clusters."""
    texts = [c["text"] for c in candidates]

    print(f"  Embedding {len(texts)} candidates...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)
    embeddings = np.array(embeddings)

    # L2-normalize for cosine-like clustering
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)

    print(f"  Clustering into {n_clusters} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=256,
        n_init=3,
    )
    labels = kmeans.fit_predict(embeddings)

    # Sample from each cluster
    rng = random.Random(seed)
    selected = []
    for cluster_id in range(n_clusters):
        members = [i for i, l in enumerate(labels) if l == cluster_id]
        if not members:
            continue
        n_sample = min(samples_per_cluster, len(members))
        sampled_indices = rng.sample(members, n_sample)
        for si in sampled_indices:
            selected.append(candidates[si])

    return selected


def build_semantic_diversity(output_dir: Path, target: int, seed: int) -> list[dict]:
    """Build the semantic diversity corpus component."""
    candidates = generate_candidate_pool(seed)
    print(f"  Generated {len(candidates)} candidate prompts from {len(TOPICS)} domains")

    # Compute cluster count and samples per cluster to hit target
    # Overshoot slightly to account for small clusters
    n_clusters = 200
    samples_per_cluster = (target * 13) // (n_clusters * 10)  # ~30% overshoot

    selected = embed_and_cluster(candidates, n_clusters, samples_per_cluster, seed)

    # Trim to target
    rng = random.Random(seed)
    if len(selected) > target:
        selected = rng.sample(selected, target)

    # Re-index
    for i, p in enumerate(selected):
        p["prompt_id"] = f"sem_{i:04d}"

    out_path = output_dir / "semantic_diversity_full.jsonl"
    _save_jsonl(selected, out_path)

    # Report domain distribution
    from collections import Counter
    domain_counts = Counter(p["domain"] for p in selected)
    print(f"  Selected {len(selected)} prompts across {len(domain_counts)} domains:")
    for d, c in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"    {d}: {c}")
    visual_count = domain_counts.get("visual_grounding", 0)
    print(f"  Visual grounding: {visual_count} (D11 target: 300-500)")

    return selected


def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build semantic diversity corpus")
    parser.add_argument("--output-dir", default="prompts")
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print("Building semantic diversity corpus...")
    build_semantic_diversity(output_dir, args.target, args.seed)


if __name__ == "__main__":
    main()
