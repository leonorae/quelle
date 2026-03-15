---
date: 2026-03-15
type: literature-review
scope: curiosity-vs-random
---

# Active Learning vs Random Selection at Low Budgets: Literature Review

## Verdict

The literature substantially resolves the question, but with important nuance. At budgets of 10–50 labeled samples, the naive answer — "uncertainty sampling beats random" — is **false**. Uncertainty/curiosity-driven methods are consistently worse than random at this scale due to the cold-start problem: models trained on so few samples produce uncalibrated, overconfident uncertainty estimates that corrupt selection. However, diversity/typicality-based methods (clustering, coreset-style coverage) can meaningfully outperform random in this regime, with gains in the range of 20–40% data reduction toward a performance threshold. The critical question for `curiosity-vs-random` is therefore more precise than originally framed: the experiment should compare **curiosity/uncertainty-driven selection specifically** against random, not diversity-driven selection — and the literature strongly predicts curiosity will lose at 10–50 samples on structured/molecular data. The experiment still has value as a domain-specific empirical confirmation (existing results are dominated by image classification, not molecular/tabular regression), but its main contribution would be a regime characterization, not a discovery. The recommendation is to **run but narrow**: define the comparison as curiosity vs. random vs. a diversity baseline (k-means or coreset), report the crossover budget, and position results as confirming or extending the cold-start literature to this domain.

---

## Key Findings

1. **Uncertainty/curiosity sampling is consistently worse than random at low budgets.** Multiple benchmark studies confirm that uncertainty-based methods underperform random selection in the <50 sample regime. The mechanism is well understood: models trained on few samples overfit and produce overconfident or uninformative uncertainty estimates, producing biased query distributions. (Hacohen et al., ICML 2022; Zhan et al., 2022; Hacohen et al. survey synthesis)

2. **A phase transition separates low-budget and high-budget regimes.** Theoretical analysis (Hacohen et al. 2022) shows that typical/dense examples should be queried at low budgets, while uncertain/atypical examples at high budgets. This predicts opposite optimal strategies. The transition point is roughly at ~1% of the total dataset size in image benchmarks, but the boundary is dataset-dependent and has not been fully characterized for molecular/tabular regression.

3. **Diversity-based methods (typicality, k-means clustering, coverage) outperform random at low budgets.** TypiClust achieves 39.4% accuracy improvement over random with just 10 labeled samples from CIFAR-10 in semi-supervised settings. Simpler k-means clustering over self-supervised features outperforms state-of-the-art AL methods at 0.2% of ImageNet. (Hacohen et al. 2022; Polosukhin et al. 2021, arXiv:2110.12033)

4. **Coreset/diversity methods for molecular data show modest but real benefits.** For machine-learned interatomic potentials (MLIPs), diversity sampling achieves 5–13% fewer labeled samples needed vs. random, with the largest gains (10.9% MAE improvement) on chemically complex systems. Workflows starting from 30 labeled samples show clear diversity advantages. (Khan et al. 2026, arXiv:2601.06916)

5. **Uncertainty-based AL for molecular property prediction shows marginal gains, often not significant.** A comprehensive evaluation of uncertainty-guided AL for aqueous solubility and redox potential prediction found improvements of only 0.02–0.14% RMSE vs. random, statistically significant only in some conditions. The benefit was larger for out-of-distribution (OOD) generalization. The authors note that practical applications requiring <250 molecule training sets are likely beyond where AL reliably helps. (Yin et al. 2023, J. Cheminformatics)

6. **Materials science regression benchmark (2025): hybrid strategies win, deep-learning AL often fails.** Across 13 small-sample regression tasks (9 materials datasets), hybrid strategies integrating uncertainty + diversity (LCMD, Tree-based-R, RD-GS) consistently outperform random. Model-free and DL-based strategies (LL4AL, MCDO) often underperform random. Median data reduction vs. random: ~22–40% at 70% R² threshold, with best cases exceeding 75%. (Bi et al. 2025, Scientific Reports)

7. **Foundation models partially dissolve the phase transition.** When large pretrained models are used as feature extractors, uncertainty sampling becomes competitive even in low-budget settings — the "phase transition" weakens or disappears. This is not applicable to the small-model regime of interest in `curiosity-vs-random`. (arXiv:2401.14555, 2024)

8. **Tabular data benchmark affirms uncertainty sampling — but on classification, not tiny-budget regression.** A dedicated tabular AL benchmark (Lu et al. 2023/2025, TMLR) finds uncertainty sampling maintains an edge on tabular classification when using a compatible model, but does not study the <50 sample regime specifically. (arXiv:2306.08954)

---

## The Cold Start Problem

Cold-start failure is the well-named phenomenon where active learning performs worse than random when the initial labeled pool is too small to support reliable scoring of candidates. The mechanism is straightforward: the model trained on 10–50 examples is poorly calibrated, leading uncertainty estimates to reflect noise and initialization artifacts rather than true informativeness. In consequence, uncertainty-based querying is biased toward outliers, rare edge cases, and regions the model happens to be uncertain about for irrelevant reasons.

This has been documented across domains:
- In deep AL on image data, it is now considered established that uncertainty sampling fails in the very-low-budget regime and requires workarounds (warm-up strategies, self-supervised pre-training, multi-stage protocols). (Hacohen et al. 2022; arXiv:2107.07724)
- In molecular property prediction specifically, Yin et al. 2023 note that "very low initial amounts of training data may inhibit the benefit of AL" and that batch sizes below 50–100 molecules show minimal improvements over random.
- In materials science regression, DL-based uncertainty methods (MCDO, LL4AL) underperform the random baseline in the small-sample settings studied by Bi et al. 2025.

The cold-start problem is distinct from the phase-transition framing: the phase transition predicts that typicality-based methods should be used at low budgets (and they work), while cold start is specifically the failure of uncertainty methods. Both analyses agree: curiosity/uncertainty at 10–50 samples is unreliable.

Proposed mitigations — clustering-based initialization, self-supervised pre-training, centroid-based pool selection, warm-up policies — all effectively convert cold-start settings into something closer to a diversity-based initial selection, sidestepping pure uncertainty scoring. These are outside the scope of a curiosity-vs-random experiment but worth noting as avenues if the domain confirms the null result.

---

## Regime Boundary

The literature does not provide a single clean budget threshold, because the boundary depends on:
- **Total dataset size**: The relevant budget measure is often the fraction of the unlabeled pool labeled. In image benchmarks, the phase transition occurs around 0.2–1% of total data. For small molecular datasets (thousands of compounds), 10–50 samples corresponds to 0.5–5%, placing it firmly in the cold-start / low-budget regime.
- **Task complexity and input dimensionality**: More complex target functions and higher-dimensional inputs push the threshold upward. Simple optimization tasks (single-peak search) allow active learning to win at very small budgets (≈11 samples in Fe-Co-Ni Kerr rotation benchmark). Multi-modal or rough landscapes delay the advantage.
- **Model type**: GP-based Bayesian optimization has better-calibrated uncertainty at low n than neural networks, so BO-style acquisition may cross the threshold at smaller budgets than NN-based AL.
- **Representation quality**: With strong pre-trained features, even uncertainty sampling can work at low budgets (2024 foundation model results). Without pre-training (the small-model regime), the cold-start range extends further.

Practical estimates from the literature:
- **Neural network AL on images**: Phase transition at ~0.5–1% labeled. For a 50k dataset, that is 250–500 samples. Below this, diversity methods dominate.
- **GP-based BO on materials**: Meaningful advantage over random is seen starting from 11–25 initial samples for simple systems; complex systems may need 50–100.
- **Molecular property prediction with GNN/MPNN**: Yin et al. 2023 suggest the practical minimum where AL shows clear benefit is 250+ molecules in the active learning pool. Below that, improvements are at best marginal.

For structured data with budgets of 10–50, the literature suggests random remains hard to beat with uncertainty/curiosity methods, and diversity methods offer at most modest gains.

---

## Molecular/Scientific Domain Specifically

The molecular and materials domain has produced the most directly relevant results for `curiosity-vs-random`.

**Molecular property prediction (GNN/MPNN):**
Yin et al. (2023, J. Cheminformatics) is the most directly applicable study. Their comprehensive evaluation of uncertainty-based AL for solubility and redox potential prediction with GNNs finds that improvements over random are statistically significant in some conditions but tiny in magnitude (0.02–0.14% RMSE), and that benefits are most pronounced for out-of-distribution generalization rather than in-distribution performance. They conclude that AL samples of 250+ molecules are needed for reliable benefit, making the 10–50 sample regime essentially below the useful threshold.

**Machine-learned interatomic potentials:**
Khan et al. (2026, arXiv:2601.06916) compare random, uncertainty, diversity, and hybrid strategies starting from 30 labeled samples with 15-sample incremental batches. Diversity sampling (k-means with farthest-point refinement) consistently matches or beats random, with 5–13% overall labeling savings and up to 10.9% MAE improvement for chemically complex Ti–O system. Uncertainty sampling shows moderate, system-dependent benefits. This is the most controlled recent comparison with a small initial pool.

**Materials optimization (GP-based BO):**
Lookman et al. (2019, npj Computational Materials) and the Oxford benchmark (Ling et al. 2022) show that Bayesian optimization-style acquisition can outperform random at very small budgets for simple target landscapes (e.g., 11 samples to reach optimum vs. >20 for random in a single-peak search). For multi-modal landscapes, the advantage narrows and can disappear. The GP uncertainty model is substantially better calibrated than neural networks at small n, which matters here.

**Small-sample regression benchmark (AutoML + AL):**
Bi et al. (2025, Scientific Reports) is the most comprehensive domain-specific result. Across 13 materials regression tasks, hybrid strategies that combine uncertainty + diversity (LCMD, RD-GS) consistently outperform random, with 20–85% data reduction at 70% R². Purely uncertainty-driven or purely DL-based methods underperform. This is regression on structured tabular data, closely matching the `curiosity-vs-random` domain.

**General pattern:** In the molecular/materials domain, curiosity/uncertainty-only methods at very small budgets show marginal or negative gains vs. random. Diversity methods or hybrid approaches show consistent but modest gains (typically 20–40% fewer samples needed to reach a performance threshold, not better absolute performance at a fixed budget).

---

## Implications for curiosity-vs-random Experiment

**Recommendation: Run, but narrow the scope and sharpen the comparison.**

The existing literature does not fully close the question for the specific setting of `curiosity-vs-random` because:
1. Most low-budget AL benchmarks focus on image classification, not structured molecular/tabular regression.
2. The experiment may use a specific curiosity definition (e.g., information-gain, epistemic uncertainty, or a custom acquisition function) that has not been benchmarked in this exact domain/model combination.
3. The crossover budget — the point at which curiosity starts outperforming random — has not been empirically mapped for the specific model architecture and dataset type in scope.

However, the literature strongly predicts the null result: at 10–50 samples on molecular/tabular regression with small models, curiosity/uncertainty selection will not outperform random selection and may be worse.

**Concrete recommendations:**

1. **Add a diversity baseline** (k-means clustering, coreset selection, or TypiClust-style typicality). Without this, the experiment will confirm the well-known result that uncertainty is bad at low budgets but miss the question of whether *any* structured selection helps. The interesting finding would be characterizing when diversity helps vs. random.

2. **Extend the budget range** beyond 50 if computationally feasible (e.g., up to 200–500 samples). The crossover budget is data-dependent and characterizing where curiosity starts working is scientifically useful.

3. **Distinguish in-distribution vs. OOD performance.** Yin et al. 2023 found that uncertainty-based AL's modest advantage is concentrated in OOD generalization. If the experiment evaluates only in-distribution test sets, it may miss the regime where curiosity provides value.

4. **Use a GP or ensemble uncertainty model if possible.** Neural network uncertainty at 10–50 samples is poorly calibrated. GP-based or ensemble uncertainty (query by committee, deep ensembles) will give a fairer test of curiosity's potential.

5. If the above modifications are not feasible, the experiment can still be run as designed to produce a domain-specific null result confirmation, which has value as empirical evidence for the `wiki/findings/` record. Frame the hypothesis as "we expect curiosity to underperform random at 10–50 samples, and aim to characterize the crossover budget."

---

## References

1. **Hacohen, G., Dekel, O., Weinshall, D.** (2022). Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets. *ICML 2022*. [https://arxiv.org/abs/2202.02794](https://arxiv.org/abs/2202.02794) — Core paper on phase transition; TypiClust; 39.4% CIFAR-10 improvement with 10 samples.

2. **Zhan, X., Wang, Q., et al.** (2022). A Comparative Survey of Deep Active Learning. *arXiv:2203.13450*. [https://arxiv.org/abs/2203.13450](https://arxiv.org/abs/2203.13450) — Benchmarks 19 DAL methods; uncertainty outperforms diversity in high-budget; no significant differences between uncertainty methods.

3. **Polosukhin, I., et al.** (2021). A Simple Baseline for Low-Budget Active Learning. *arXiv:2110.12033*. [https://arxiv.org/abs/2110.12033](https://arxiv.org/abs/2110.12033) — K-means clustering outperforms sophisticated AL at low budgets (0.2% ImageNet).

4. **Lu, P., et al.** (2023/2025). An Expanded Benchmark that Rediscovers and Affirms the Edge of Uncertainty Sampling for Active Learning in Tabular Datasets. *TMLR 2025*. [https://arxiv.org/abs/2306.08954](https://arxiv.org/abs/2306.08954) — Uncertainty sampling edge on tabular classification; compatible model required.

5. **Yin, T., Panapitiya, G., Coda, E.D., et al.** (2023). Evaluating uncertainty-based active learning for accelerating the generalization of molecular property prediction. *Journal of Cheminformatics*, 15, 105. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10633997/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10633997/) — Comprehensive molecular AL evaluation; improvements 0.02–0.14% RMSE; practical minimum ~250 molecules.

6. **Bi, J., Xu, Y., Conrad, F., Wiemer, H., Ihlenfeldt, S.** (2025). A comprehensive benchmark of active learning strategies with AutoML for small-sample regression in materials science. *Scientific Reports*. [https://www.nature.com/articles/s41598-025-24613-4](https://www.nature.com/articles/s41598-025-24613-4) — 13 materials regression tasks; hybrid strategies 20–85% data reduction vs. random; DL-based uncertainty underperforms.

7. **Khan, M.A., D'Souza, A., Choyal, V.** (2026). Active Learning Strategies for Efficient Machine-Learned Interatomic Potentials Across Diverse Material Systems. *arXiv:2601.06916*. [https://arxiv.org/abs/2601.06916](https://arxiv.org/abs/2601.06916) — Starts from 30 labeled samples; diversity sampling 5–13% labeling savings; 10.9% MAE improvement for Ti–O.

8. **Ling, J., et al.** (2022). Benchmarking Active Learning Strategies for Materials Optimization and Discovery. *Oxford Open Materials Science*, 2(1), itac006. [https://academic.oup.com/ooms/article/2/1/itac006/6637521](https://academic.oup.com/ooms/article/2/1/itac006/6637521) — Fe-Co-Ni benchmark; Expected Improvement best overall; AL outperforms random from as few as 11 samples on simple landscapes.

9. **Settles, B.** (2009). Active Learning Literature Survey. *University of Wisconsin–Madison, TR1648*. [https://burrsettles.com/pub/settles.activelearning.pdf](https://burrsettles.com/pub/settles.activelearning.pdf) — Foundational survey; uncertainty sampling framework; structured instances.

10. **Lookman, T., et al.** (2019). Active learning in materials science with emphasis on adaptive sampling using uncertainties for targeted design. *npj Computational Materials*. [https://www.nature.com/articles/s41524-019-0153-8](https://www.nature.com/articles/s41524-019-0153-8) — GP-based AL for materials; uncertainty-guided experimental design framework.

11. **Barata, R., et al.** (2021). Active learning for imbalanced data under cold start. *arXiv:2107.07724*. [https://arxiv.org/abs/2107.07724](https://arxiv.org/abs/2107.07724) — Cold-start AL under class imbalance; multi-stage warmup strategies.

12. **Revisiting Active Learning in the Era of Vision Foundation Models** (2024). *arXiv:2401.14555*. [https://arxiv.org/html/2401.14555v2](https://arxiv.org/html/2401.14555v2) — Foundation model features dissolve phase transition; uncertainty competitive even at low budgets with strong pre-training. (Note: not applicable to small-model regime.)

13. **MADE: Benchmark Environments for Closed-Loop Materials Discovery** (2026). *arXiv:2601.20996*. [https://arxiv.org/abs/2601.20996](https://arxiv.org/abs/2601.20996) — Closed-loop materials discovery benchmark; dynamic evaluation of AL strategies under limited oracle budget.
