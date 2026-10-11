# Model card — NavDiag local v0.3

The primary branch is satellite LOS/NLOS classification on KLT and UrbanNav. The current experiments are paired same-domain/area holdouts on KLT, frozen source transfer to UrbanNav, and six UrbanNav district-disjoint directions. Five neural models plus three traditional references are compared; validation controls selection. See [current results](experiments/urbannav_generalization.md), [protocols](experiments/nlos_protocols.md) and [project review](PROJECT_REVIEW.md). Cross-city performance and positioning benefits remain unverified.

The first sections below describe the separate historical positioning/alarm branch. Satellite-classification details and chronological development records follow; their datasets, labels and metrics must not be interchanged.

## Task and intended use

Predict whether the current GNSS baseline horizontal error exceeds a configurable threshold (default 10 m). This default is a coarse learning/demo alarm definition, with no application-specific requirement established; it is not an achieved or acceptable positioning accuracy. Continuous Median/P95/RMSE and within-1/2/5/10 m fractions separately assess positions. This is a research/learning pipeline, not a navigation safety or integrity monitor. No real deployment performance has been established.

## Observations and labels

Allowed features: satellite count, mean/minimum C/N0, unweighted geometry HDOP, WLS pseudorange residual RMS and causal estimated speed. Trip IDs, timestamps, reference coordinates and reference errors do not become learned input features. Speed uses current/previous WLS positions with a 70 m/s cap. Ground truth is used only to create labels and evaluate trajectories.

GSDC 2021 derived examples lack C/N0. Unavailable measurements remain NaN. Features with less than 50% training coverage are dropped; remaining median imputation and standardization are fitted only on training data. The fixed rule treats an unavailable C/N0 comparison as false, so its signal-based branch is unavailable on these files.

## Models and selection

- Fixed thresholds on C/N0, HDOP and residuals; its binary score is not a probability.
- Median imputation, StandardScaler and balanced logistic regression.
- Median imputation and balanced RF: 150 trees, depth 8, minimum leaf 6.
- Optional unidirectional GRU: hidden size 24, 10 past/current epochs, weighted BCE, Adam, validation early stopping. Windows reset across trips, devices and gaps over 5 seconds. Ten epochs are not necessarily ten seconds.

Whole trips, including all phones from one drive, remain in one split. The default random split is approximately 60/20/20; a complete explicit manifest can override it. Thresholds maximize validation F1 on a fixed grid; the model with highest validation F1 wins, even if it is the rule baseline. No test metric selects a model or threshold. Repeated routes and devices still require separate generalization experiments.

## Positioning and filtering

The parser uses supplied satellite states and applies satellite clock, inter-signal, ionospheric and tropospheric corrections. It does not derive ephemerides or perform carrier-phase/RTK processing. Least squares estimates ECEF position and one clock per constellation, with Earth rotation correction. Huber IRLS provides a robust comparison; failures fall back to ordinary WLS and are counted. Reference ECEF positions are interpolated only within support and the configured maximum gap.

The causal planar Kalman baseline estimates east/north position and velocity, resets across large gaps and rejects extreme innovations. Fixed observation sigma/process noise are selected by validation RMSE. Adaptive observation variance uses monotone risk bins and validation reference errors; it is an empirical mapping, not probability calibration. Both are then frozen for testing. The filter accepts optional calibrated, gravity-removed acceleration already expressed in ENU. Only the simulator supplies this input in the included experiments; public GSDC runs use no IMU. Raw IMU calibration, attitude estimation and a full strapdown INS are outside v0.2.

## Evaluation and limitations

Classification metrics include Precision, Recall, F1, false-positive rate, AP, ROC AUC, Brier score and `[[TN, FP], [FN, TP]]`. Recall/F1 are null if no reference positives exist; Precision is null if no positive prediction exists. AP/AUC are null for a single label class. Risk scores are not calibrated probabilities. Aggregate metrics weight epochs equally rather than trips equally.

After the 2026-10-11 GSDC 2021 timestamp/reference-height correction, the public experiment contains 2,824 training epochs, 1,737 validation epochs and 11 test epochs from parser examples. The previous real validation F1 (GRU 0.991) is invalid and withdrawn because the old parser generated incorrect labels. Corrected validation has only five >10 m positives; selected GRU validation F1 is 0.308 and the test has zero positives. This checks an engineering path, not representative detection performance. See [correction record](experiments/gsdc2021_correction.md). Device, year and route shift are confounded; more complete independent drives are needed.

The navigation simulation contains 15 generated trips. Its observation geometry, biases, signal statistics and acceleration are simplified. Real WLS computation on simulated pseudoranges does not make the experiment real data. In this simulation fixed filtering outperformed adaptive filtering; in the 11-epoch real smoke test adaptive filtering also did not improve WLS RMSE. No consistent learned-filter improvement has been established.

Signal statistics and model risk alone cannot establish an NLOS cause. Diagnostics must distinguish recorded facts from hypotheses and missing evidence. The optional LLM adapters have protocol tests; no live API quality evaluation is claimed without configured credentials.

## Separate satellite LOS/NLOS task (local v0.3)

The separate local satellite LOS/NLOS branch consumes the pinned Fossett Laboratory numeric table derived from KLT: 26,522 GPS/BeiDou observations, four short runs on 2023-11-09, with fisheye sky-mask labels. Five published radio/geometry features are consumed; raw RINEX extraction and camera-label accuracy are not independently reproduced. Scene IDs and reference labels are excluded from model inputs; missing-value imputation is fitted on training only.

Two KLT2 runs supply 9,830 training rows, KLT1 supplies 6,607 validation rows, and KLT3 supplies 10,085 test rows. Areas are disjoint. Current comparisons include five neural classifiers (MLP, temporal CNN, LSTM, GRU, Transformer), plus C/N0 < 30, LR and RF. Constant LOS/NLOS sanity baselines remain in JSON reports and are hidden in the UI. No component ablations are included. The validation-selected RF test Precision/Recall/F1 are 0.672/0.778/0.721. Neural test F1 values are 0.705/0.719/0.708/0.710/0.712, respectively; the validation-preferred neural model is GRU. Model selection remains based on validation, not the best test score. See [five-network experiment](experiments/five_neural_nlos.md).

All five neural networks consume the same 32 past/current satellite observations with train-only imputation/scaling and the same AdamW/BCE/early-stopping budget. The MLP flattens history; the CNN uses left-padded convolutions and mean/last-token summaries; LSTM/GRU are two-layer unidirectional networks of hidden size 64. Different parameter counts and inductive biases remain, and no architecture-specific hyperparameter search has been completed. One seed and one held-out area do not establish a statistically reliable ranking. LR/RF consume current features, so neural-versus-tree comparisons include different history access.

The Transformer uses the same five features over 32 past/current observations of one satellite, resetting across trips, receivers and gaps >5 seconds. Short history repeats the first available observation on the left. A 64-wide, four-head, two-layer encoder with causal attention, sinusoidal positions and a last-token classifier has 67,521 parameters. Train-only median imputation/scaling, weighted BCE and AdamW are used. Validation BCE selects the checkpoint (best epoch 1 of 7 run, patience 6), then validation F1 selects threshold 0.65. Unknown labels may supply observed history but never supervision. This compares complete pipelines with different history access, not an isolated architecture advantage. See [Transformer experiment](experiments/transformer_nlos.md).

The selected RF has 1,366 false positives and 801 false negatives. At C/N0 in (35,45] dB-Hz it finds only 67 of 395 NLOS labels. Per-system, C/N0 and elevation metrics expose these failures without retuning on test. A single test area cannot support independent-drive confidence intervals or city/device generalization.

LOS may include multipath. Building/tree/vehicle labels and completely untracked satellites are absent; object labels remain unknown. This is not an object detector or proof of improved positioning, and KLT is not joined to unrelated GSDC samples. Further tuning needs an additional untouched final holdout. See [labels](OBSTRUCTION_LABELS.md), [experiment](experiments/v0_3_nlos.md), and [reuse](OPEN_SOURCE_REFERENCES.md).

## Reproducibility

The KLT experiment is a paired in-domain/cross-region comparison across three target areas, with nine acquisitions, three dates, 2,851 epochs and 45,228 observations. Each pair shares validation/test acquisitions; cross-region training removes all target-area training acquisitions. Fixed original defaults are used, without continuing the earlier hyperparameter search. In-domain training fractions are 59–65%; acquisition integrity takes precedence over exact 6:2:2 ratios. See [protocol design and results](experiments/nlos_protocols.md).

All nine acquisitions are re-extracted with the same pinned RTKLIB/Fossett pipeline. The four historical acquisitions match all original labels and numeric features within 1e-6, including missingness. New camera-label times match receiver epochs one-to-one within 50 ms; the June 2021 receiver has a roughly 5 ms fractional offset. KLT3-649 is excluded as unreliable. The raw KLT3-404 label file contains 2,461 NLOS observations rather than the source README's 2,391; actual source labels are preserved. The optional standalone GPL preprocessing adapter is outside the MIT main package.

Equal-weight target-area means describe observed model advantages; no test score chooses hyperparameters or checkpoints. Three nearby areas and one seed do not establish statistically stable rankings. Some test acquisitions were inspected during earlier development. Removing target-area training changes both area exposure and sample size; same-day adjacent acquisitions and date/illumination differences remain limitations. This is exploratory comparison, not a new untouched final benchmark.

The latest bounded search uses eight predefined candidates per LR/RF/neural family (56 total), including each default. Trial records contain validation scores only; selected configurations are frozen to disk before final refitting and test evaluation. Validation selects CNN (F1 0.7637), with development-holdout test F1 0.7215 against the original RF 0.7212. The tiny difference is not evidence of stable improvement. LSTM test F1 is 0.7243 but it is not selected by test rank. Tuned RF improves validation yet declines on test, illustrating selection/generalization limits. The current test area has been repeatedly inspected during development; further claims need a new untouched final holdout. See [search record](experiments/nlos_hyperparameter_search.md).

Reports record dataset SHA-256, seed, features/coverage, exact trip memberships, validation thresholds, history and metrics. Public source URLs are commit-pinned with size and SHA-256 in `config/public_sources.json`; preprocessing records alignment/drop/fallback counts. `requirements-lock.txt` is the tested Python 3.12 dependency snapshot excluding platform-specific PyTorch; CPU PyTorch version is installed separately as documented. This is not a cross-platform hash lock.

Published result summaries and limitations are in `docs/experiments/v0_2.md`. Third-party raw data and model binaries are excluded from Git. Load only model files you generated or trust, since joblib is a pickle-based format.

C/N0 zero placeholders (927 rows) are converted to NaN. The local adapter recomputes C/N0 standard deviation over five trailing per-satellite observations, requiring two observed values and resetting after gaps above five seconds. Other published features are retained.

## Unseen UrbanNav districts

The fixed KLT source bundle is evaluated without changing validation thresholds on TST, WP and MK, followed by six whole-drive train/validation/test district permutations inside UrbanNav. There are 5,671 raw epochs and 77,744 satellite rows, but only 27,839 accepted static-building reference labels. These references use independent SPAN positions and building masks, not classifier features; dynamic obstacles and uncertain geometry remain unknown. MK coverage is 19.9%. Source/target antenna, acquisition dates and camera/map reference methods differ; within UrbanNav the antenna and reference method match, but dates remain confounded with areas. Three drives in Hong Kong do not establish cross-city performance. See [results](experiments/urbannav_generalization.md) and [measured domain audit](experiments/nlos_domain_audit.md).

A fixed C/N0 rule must have identical scores when paired protocols share the same test observations and labels; this invariance does not diagnose geographic distribution shift. KLT C/N0 distributions are close, while UrbanNav class-conditional distributions differ in the observed accepted references. Distribution distances are descriptive, without independent-drive significance claims. No further tuning was performed.
