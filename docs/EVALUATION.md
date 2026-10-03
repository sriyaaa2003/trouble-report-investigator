# Evaluation

Two benchmarks on public Mozilla Core bug reports, both with a strict time split. Raw outputs are in
[`results/`](../results). Reproduce with:

```bash
investigator fetch                     # downloads the data (about 25 minutes, resumable)
investigator eval-dups --out results/dups.json
investigator eval-components --out results/components.json
```

## Data

Bug reports of the **Core** product of Mozilla's public Bugzilla, created 2021-2024, fetched through the
official REST API. The tracker had about 54,800 resolved bugs (FIXED or DUPLICATE) in that window. I
sampled (seed 13) up to 3,000 duplicates and 9,000 fixed bugs, then added every master that a sampled duplicate
points at, and kept those whose description could be downloaded (restricted security bugs cannot).

| | |
|---|---|
| Reports in the corpus | 14,037 |
| of which resolved as DUPLICATE | 3,122 |
| Distinct components (subsystem labels) | 143 |
| Text used | summary + first comment (boilerplate removed) + crash signature if any |
| Recorded fix | the comment that records a landed change, kept only for FIXED reports |

References to other bugs ("bug 1234567", Bugzilla URLs) are removed from all text so a system cannot read the
answer off the report.

## Benchmark 1: finding earlier reports of the same problem

**Task.** Given a report that was later closed as a duplicate, retrieve an earlier report of the same issue.
Ground truth comes from human triagers: the master it was marked a duplicate of, plus the master's other
duplicates and chained masters, created before the query.

**Protocol.**
- The corpus for each query contains only reports **created before** that query.
- Fusion weight chosen on development queries from 2023 (431 queries); every number below is on the
  **399 held-out queries from 2024**, searched against the 14,037-report corpus.
- 95% intervals are bootstrap over queries; differences are paired (same queries).

**Systems.** BM25 over a code-aware tokenizer; dense cosine with `BAAI/bge-small-en-v1.5`; reciprocal-rank
fusion (RRF) of the two; and a z-score weighted fusion with the dense weight picked on 2023 (0.85).

| System | recall@1 | recall@5 | recall@10 | recall@20 | MRR |
|---|---|---|---|---|---|
| BM25 | 0.501 [0.451, 0.549] | 0.669 [0.622, 0.714] | 0.709 [0.662, 0.754] | 0.769 [0.727, 0.810] | 0.577 [0.533, 0.619] |
| dense | 0.561 [0.514, 0.609] | 0.717 [0.672, 0.759] | 0.784 [0.739, 0.825] | 0.855 [0.817, 0.887] | 0.636 [0.593, 0.676] |
| hybrid (RRF) | 0.546 [0.499, 0.594] | 0.742 [0.697, 0.782] | 0.789 [0.747, 0.827] | 0.842 [0.805, 0.875] | 0.635 [0.593, 0.676] |
| **hybrid (z-score, w_dense=0.85)** | **0.607 [0.556, 0.652]** | **0.759 [0.717, 0.799]** | **0.830 [0.789, 0.865]** | **0.875 [0.840, 0.905]** | **0.682 [0.640, 0.721]** |

Paired difference against BM25:

| System | recall@1 | recall@5 | recall@10 | recall@20 | MRR |
|---|---|---|---|---|---|
| dense | 0.060 [0.008, 0.110] | 0.048 [0.000, 0.095] | 0.075 [0.030, 0.123] | 0.085 [0.045, 0.133] | 0.059 [0.016, 0.101] |
| hybrid (RRF) | 0.045 [0.010, 0.083] | 0.073 [0.038, 0.108] | 0.080 [0.045, 0.115] | 0.073 [0.038, 0.110] | 0.058 [0.030, 0.085] |
| hybrid (z-score) | 0.105 [0.065, 0.143] | 0.090 [0.055, 0.125] | 0.120 [0.083, 0.158] | 0.105 [0.070, 0.145] | 0.105 [0.075, 0.134] |

**What this shows.**
- A tuned lexical + semantic hybrid finds an earlier report of the same issue in the top 10 for about 83% of
  held-out duplicates, versus about 71% for BM25 alone. The gain is +0.12 recall@10 and the interval excludes zero.
- Dense alone beats BM25, but only narrowly at recall@5 (the interval touches zero). Lexical and semantic
  signals are complementary, which is why combining them wins.
- The fusion method matters: plain RRF is clearly better than BM25 but worse than the weighted fusion at
  rank 1. The dense weight picked on the development year (0.85) sits at the edge of the grid I tried
  (0.3, 0.5, 0.7, 0.85), so a better value may exist; I did not search further.
- About 1 in 6 duplicates is still not in the top 10, and about 1 in 8 not in the top 20.

## Benchmark 2: predicting the affected subsystem

**Task.** Predict the component a report ends up filed under, from its text.

**Protocol.**
- Train on reports created before 2024; test on reports created from 2024 (2,689 test reports).
- Only components with at least 40 training examples are modelled: **64 classes**. Reports in rarer
  components (14% of the test set) are excluded, so these numbers describe the common components.
- Ensemble weights chosen on 2023 (models refit on earlier data), then refit on everything before 2024 and
  evaluated once on 2024.

| System | top-1 accuracy | top-3 accuracy | right broad area* | macro-F1 |
|---|---|---|---|---|
| always the most common component | 0.090 [0.080, 0.100] | 0.218 [0.203, 0.234] | 0.208 | 0.003 |
| TF-IDF + logistic regression | 0.593 [0.574, 0.612] | 0.800 [0.785, 0.815] | 0.697 | 0.518 |
| embedding + logistic regression | 0.510 [0.491, 0.529] | 0.747 [0.731, 0.763] | 0.639 | 0.435 |
| similar-case vote (k=10) | 0.492 [0.473, 0.511] | 0.705 [0.688, 0.721] | 0.599 | 0.420 |
| **ensemble** (weights 2.0 / 0.5 / 0.5) | **0.606 [0.588, 0.625]** | **0.822 [0.807, 0.836]** | **0.714** | **0.526** |

\* The prediction has the same parent area as the truth, i.e. the text before the colon
(`Graphics: WebRender` and `Graphics` count as the same area).

Paired difference against the TF-IDF classifier:

| System | top-1 | top-3 |
|---|---|---|
| embedding + LR | -0.083 [-0.100, -0.064] | -0.052 [-0.068, -0.037] |
| similar-case vote | -0.101 [-0.120, -0.083] | -0.095 [-0.111, -0.079] |
| ensemble | +0.013 [0.004, 0.023] | +0.022 [0.013, 0.032] |

**What this shows.**
- The correct component is the first suggestion about 61% of the time and among the top three about 82%
  of the time, against 9% and 22% for always guessing the most common one.
- **A plain TF-IDF classifier does most of the work.** The ensemble improves on it by only 1.3 points
  top-1 (2.2 top-3). That improvement is statistically real but small, and I would not call it important.
- Embeddings and the similar-case vote are *worse* than TF-IDF on their own for this task. The vote is
  included because it makes a prediction explainable (it points at concrete earlier reports), not because
  it is more accurate.
- Even the broad area is wrong about 29% of the time. Many errors are between related components, which I
  did not analyse further.

## Threats to validity (read these before quoting any number)

- **Sampled corpus.** The corpus is about a quarter of resolved Core bugs in the window plus the masters of
  sampled duplicates. The real tracker has more near-duplicate distractors, so recall on the full tracker
  would be lower than reported.
- **Ground truth is human and imperfect.** A retrieved report that is truly a duplicate but was never
  linked counts as a miss; component labels are the final component after triage and can be noisy.
- **One project, one organisation, four years.** Nothing here shows the numbers would hold for another
  product, another company's reports, or Cloud RAN Layer 1 trouble reports (see
  [APPLYING_TO_CLOUD_RAN_L1.md](APPLYING_TO_CLOUD_RAN_L1.md)).
- **Small models, no tuning beyond one weight.** I used a small embedding model and tuned only the fusion
  weight and ensemble weights. A cross-encoder reranker, fine-tuning, or a larger embedding model were not tried.
- **A few reports are newer than the study window.** About 60 corpus reports were created in 2025-2026
  (most likely masters that older duplicates point at). They fall in the component test period, which starts
  on 2024-01-01 and has no upper bound.
- **Not evaluated:** the investigation steps and evidence displays (judged only by reading examples), and
  the optional LLM hypothesis step (never run against a live model).
