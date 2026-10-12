# Model card — NavDiag local v0.3

The primary branch is satellite LOS/NLOS classification on KLT and UrbanNav. The current experiments are paired same-domain/area holdouts on KLT, frozen source transfer to UrbanNav, and six UrbanNav district-disjoint directions. Five neural models plus three traditional references are compared; validation controls selection. See [current results](experiments/urbannav_generalization.md), [protocols](experiments/nlos_protocols.md) and [project review](PROJECT_REVIEW.md). Cross-city performance and positioning benefits remain unverified.

Updated 2026-10-12: the separate positioning/alarm branch now includes three full UrbanNav drives and a Doppler-velocity KF comparison. GSDC parser samples, the four-run KLT comparison and tuning results below remain dated development records. Their datasets, labels and metrics must not be interchanged. Learning guidance is in [LEARNING_PATH](LEARNING_PATH.md).

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

The historical GSDC adapter uses supplied satellite states and satellite clock, inter-signal, ionospheric and tropospheric corrections; it does not derive ephemerides or perform carrier-phase/RTK processing. Its own LS estimates ECEF and one clock per constellation, with Earth rotation correction; Huber failures fall back to WLS and are counted. GSDC reference interpolation remains within support and the configured maximum gap. The current UrbanNav standalone adapter instead calls RTKLIB broadcast single-frequency GPS/BeiDou SPP, with a 10-degree mask and independent SPAN time/quality scoring guards.

The causal planar Kalman baseline estimates east/north position and velocity, resets across large gaps and rejects extreme innovations. Fixed observation sigma/process noise are selected by validation RMSE. Adaptive observation variance uses monotone risk bins and validation reference errors; it is an empirical mapping, not probability calibration. Both are then frozen for testing. The filter accepts optional calibrated, gravity-removed acceleration already expressed in ENU. Only the simulator supplies this input in the included experiments; public GSDC runs use no IMU. Raw IMU calibration, attitude estimation and a full strapdown INS are outside v0.2.

## Evaluation and limitations

Classification metrics include Precision, Recall, F1, false-positive rate, AP, ROC AUC, Brier score and `[[TN, FP], [FN, TP]]`. Recall/F1 are null without reference positives; Precision is null without positive predictions; AP/AUC are null for a single class. Risk scores are not calibrated. Individual reports weight scored observations equally; KLT/UrbanNav regional summaries and the positioning rotation use explicitly declared equal-area/equal-drive averages.

After the 2026-10-11 GSDC 2021 timestamp/reference-height correction, the public experiment contains 2,824 training epochs, 1,737 validation epochs and 11 test epochs from parser examples. The previous real validation F1 (GRU 0.991) is invalid and withdrawn because the old parser generated incorrect labels. Corrected validation has only five >10 m positives; selected GRU validation F1 is 0.308 and the test has zero positives. This checks an engineering path, not representative detection performance. See [correction record](experiments/gsdc2021_correction.md). Device, year and route shift are confounded; more complete independent drives are needed.

The historical navigation simulation contains 15 simplified generated trips. Running LS on synthetic pseudoranges does not make it real data. Fixed filtering outperformed adaptive filtering there. In the corrected GSDC 11-point sample, adaptive RMSE was 6.383 m versus WLS 6.391 m; this 0.008 m difference is not reliable evidence of benefit. Current full-drive results remain mixed, as recorded under Full-drive positioning and alarm expansion.

Signal statistics and model risk alone cannot establish an NLOS cause. Diagnostics distinguish facts, hypotheses and missing evidence. A small 17-request live GLM-5.3 evidence-answer audit is recorded in [quality checks](experiments/glm_quality.md); failures and revisions are preserved. It is not a blind or representative answer-correctness evaluation, and tool-Agent generation quality remains unverified.

## Separate satellite LOS/NLOS task (local v0.3)

This section records the historical four-run KLT comparison: 26,522 GPS/BeiDou observations on 2023-11-09 from the pinned Fossett numeric table, with fisheye sky-mask labels. Later paired main experiments re-extract nine runs as described under Reproducibility. Camera-label accuracy is not independently established. Scene IDs and reference labels are excluded from inputs; imputation fits training only.

In the historical four-run comparison, KLT2 supplies 9,830 train rows, KLT1 6,607 validation rows and KLT3 10,085 test rows. Five neural classifiers plus C/N0, LR and RF are compared, with no component ablations. Historical RF Precision/Recall/F1 are 0.672/0.778/0.721; neural F1 values are 0.705/0.719/0.708/0.710/0.712 and GRU is validation-preferred among networks. Current paired/UrbanNav results are recorded separately. Constant sanity baselines remain in reports but are hidden in the UI. See [historical five-network experiment](experiments/five_neural_nlos.md).

All five neural networks consume the same 32 past/current satellite observations with train-only imputation/scaling and the same AdamW/BCE/early-stopping budget. The MLP flattens history; the CNN uses left-padded convolutions and mean/last-token summaries; LSTM/GRU are two-layer unidirectional networks of hidden size 64. Different parameter counts and inductive biases remain, and no architecture-specific hyperparameter search has been completed. One seed and one held-out area do not establish a statistically reliable ranking. LR/RF consume current features, so neural-versus-tree comparisons include different history access.

The Transformer uses five features over 32 past/current observations of one satellite, resetting across trips, receivers and gaps >5 seconds. Short history repeats the first observation. A 64-wide, four-head, two-layer causal encoder with sinusoidal positions and last-token classification has 67,521 parameters. Train-only imputation/scaling, weighted BCE and AdamW are used. In the historical four-run experiment, the best checkpoint was epoch 1 of 7 and the validation threshold was 0.65; these are not current per-fold settings. Unknown labels may supply observed history but never supervision. See [historical Transformer experiment](experiments/transformer_nlos.md).

The historical four-run selected RF had 1,366 false positives and 801 false negatives; in C/N0 (35,45] dB-Hz it found only 67 of 395 NLOS labels. These counts belong to that development test, not current regional totals. Per-system/C/N0/elevation slices expose failures without retuning. A single test area cannot support independent-drive confidence intervals or city/device generalization.

LOS may include multipath. Building/tree/vehicle labels and completely untracked satellites are absent; object labels remain unknown. This is not an object detector or proof of improved positioning, and KLT is not joined to unrelated GSDC samples. Further tuning needs an additional untouched final holdout. See [labels](OBSTRUCTION_LABELS.md), [experiment](experiments/v0_3_nlos.md), and [reuse](OPEN_SOURCE_REFERENCES.md).

## Reproducibility

The KLT experiment is a paired in-domain/cross-region comparison across three target areas, with nine acquisitions, three dates, 2,851 epochs and 45,228 observations. Each pair shares validation/test acquisitions; cross-region training removes all target-area training acquisitions. Fixed original defaults are used, without continuing the earlier hyperparameter search. In-domain training fractions are 59–65%; acquisition integrity takes precedence over exact 6:2:2 ratios. See [protocol design and results](experiments/nlos_protocols.md).

All nine acquisitions are re-extracted with the same pinned RTKLIB/Fossett pipeline. The four historical acquisitions match all original labels and numeric features within 1e-6, including missingness. New camera-label times match receiver epochs one-to-one within 50 ms; the June 2021 receiver has a roughly 5 ms fractional offset. KLT3-649 is excluded as unreliable. The raw KLT3-404 label file contains 2,461 NLOS observations rather than the source README's 2,391; actual source labels are preserved. The optional standalone GPL preprocessing adapter is outside the MIT main package.

Equal-weight target-area means describe observed model advantages; no test score chooses hyperparameters or checkpoints. Three nearby areas and one seed do not establish statistically stable rankings. Some test acquisitions were inspected during earlier development. Removing target-area training changes both area exposure and sample size; same-day adjacent acquisitions and date/illumination differences remain limitations. This is exploratory comparison, not a new untouched final benchmark.

The historical bounded search used eight predefined candidates per family (56 total). Validation selected CNN (F1 0.7637), with development-holdout test F1 0.7215 against RF 0.7212. This tiny difference is not stable improvement; the holdout was already repeatedly inspected. This search remains stopped and its winners are not transferred into current fixed-default main experiments. Full trials and other model outcomes remain in the [historical search record](experiments/nlos_hyperparameter_search.md).

Reports record dataset SHA-256, seed, features/coverage, exact trip memberships, validation thresholds, history and metrics. Public source URLs are commit-pinned with size and SHA-256 in `config/public_sources.json`; preprocessing records alignment/drop/fallback counts. `requirements-lock.txt` is the tested Python 3.12 dependency snapshot excluding platform-specific PyTorch; CPU PyTorch version is installed separately as documented. This is not a cross-platform hash lock.

Published current results are in [paired protocols](experiments/nlos_protocols.md), [UrbanNav NLOS](experiments/urbannav_generalization.md) and [full-drive positioning](experiments/positioning_expansion.md); v0.2 remains a historical record. Third-party raw data and binaries are excluded from Git. Load only model files you generated or trust, since joblib is pickle-based.

C/N0 zero placeholders (927 rows) are converted to NaN. The local adapter recomputes C/N0 standard deviation over five trailing per-satellite observations, requiring two observed values and resetting after gaps above five seconds. Other published features are retained.

## Unseen UrbanNav districts

The fixed KLT source bundle is evaluated without changing validation thresholds on TST, WP and MK, followed by six whole-drive train/validation/test district permutations inside UrbanNav. There are 5,671 raw epochs and 77,744 satellite rows, but only 27,839 accepted static-building reference labels. These references use independent SPAN positions and building masks, not classifier features; dynamic obstacles and uncertain geometry remain unknown. MK coverage is 19.9%. Source/target antenna, acquisition dates and camera/map reference methods differ; within UrbanNav the antenna and reference method match, but dates remain confounded with areas. Three drives in Hong Kong do not establish cross-city performance. See [results](experiments/urbannav_generalization.md) and [measured domain audit](experiments/nlos_domain_audit.md).

A fixed C/N0 rule must have identical scores when paired protocols share the same test observations and labels; this invariance does not diagnose geographic distribution shift. KLT C/N0 distributions are close, while UrbanNav class-conditional distributions differ in the observed accepted references. Distribution distances are descriptive, without independent-drive significance claims. No further tuning was performed.

## Full-drive positioning and alarm expansion

The exploratory rotation evaluates each of TST, WP and MK once, with fixed train/validation roles: 5,671 raw epochs over about 94.5 minutes, 1,193 reference-scored fixes, only three acquisitions. The primary split trains TST, validates WP and tests MK. MK has 3,350 raw epochs, 577 successful SPP fixes, 1,270 quality-eligible references and 139 scored fixes. Its raw SPP availability is 17.2%; reference coverage is partial and the longest solution outage is about 464 seconds. Previously explored data and area/date confounding preclude an untouched or cross-city claim.

All successful observation epochs enter causal filtering and GRU history; reference guards only select supervision and scoring rows. The origin is the first successful SPP fix. Failed SPP epochs are not credited with a valid position or alert output. The primary MK alert winner is validation-selected RF, threshold 0.25, test Precision/Recall/F1 0.529/0.818/0.643 on 139 scored fixes; this is baseline error >10 m, not satellite NLOS or post-filter alarm quality.

The added causal planar Doppler KF consumes RTKLIB velocity/qv rotated into ENU. It approximates position/velocity correlation as zero; velocity covariance is not empirically calibrated. Existing sigma/process candidates are selected only on each validation drive. It keeps the same scored population and does not retrain alert/NLOS models. This is a method-inspired baseline, not TDCP, FGO or full GNSS/IMU reproduction.

Test RMSE for SPP/Doppler is 7.881/8.531 m on TST, 12.153/11.170 m on WP and 15.481/14.610 m on MK. MK Doppler Median/P95 worsen despite lower RMSE. Accuracy is conditional on reference-scored successful fixes; joint availability/accuracy uses all quality-eligible reference epochs. Full denominators, parameters, provenance and failures are in [expanded results](experiments/positioning_expansion.md) and [method sources](research/positioning_methods.md).
