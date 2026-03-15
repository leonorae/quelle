---
title: "Bisimulation Probe Positioning"
date: 2026-03-15
experiments: [behavioral-projections]
type: novelty-review
---

# Bisimulation Probe Prior Art

## Mathematical Ancestor: Structural Probes

Hewitt & Manning (NAACL 2019) — same framework, different target distance. They
learn a linear probe B such that d_B(h_i, h_j) approximates syntactic tree distance
between tokens i and j. The bisimulation probe uses the same mechanism but targets
behavioral distance (how similarly the model treats two representations downstream)
rather than syntactic distance.

Must cite prominently. The contribution is the choice of target metric, not the
probe architecture.

## Conceptual Ancestor: Deep Bisimulation for Control

Zhang et al. (ICLR 2021) — same concept (representation distance should equal
behavioral distance), but learns representations during training rather than
analyzing them post-hoc. They train an encoder whose latent distances match
bisimulation distances for RL policies. The bisimulation probe applies this idea
as an analysis tool on a frozen model.

## Motivation: The Representational-Functional Gap

- **Klabunde et al. survey** — identifies the gap between representational
  similarity metrics (CKA, CCA) and functional similarity. The probe sits exactly
  in this gap: learned from representations, targeted at function.
- **Ding et al. (NeurIPS 2021)** — show existing metrics correlate weakly with
  functional differences. Motivates a learned bridge rather than a fixed statistical
  summary.

## Strongest Novelty Claim

Identifying the **behaviorally relevant subspace** — not just measuring correlation
between representation similarity and behavioral similarity, but learning the
specific linear projection under which they align. The probe's B matrix defines
this subspace; its eigenspectrum reveals dimensionality and structure.

## Framing

Method novelty, not phenomenon novelty. The gap between representational and
functional similarity is well-documented. The contribution is a formal method
(learned linear probe targeting bisimulation distance) that provides analytical
leverage on this recognized problem.

Source: novelty review, March 2026.
