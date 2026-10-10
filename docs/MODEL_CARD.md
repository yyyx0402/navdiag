# Model card — NavDiag v0.2

## Task and intended use

Predict whether the current GNSS baseline horizontal error exceeds a configurable threshold (default 10 m). This is a research/learning pipeline, not a navigation safety or integrity monitor. It produces risk scores and offline diagnostic evidence; no real deployment performance has been established.

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

The public experiment contains 2,825 training epochs from one drive, 1,739 validation epochs from another and 11 test epochs from two parser examples. This split checks the engineering path, not representative generalization. The test has zero unreliable epochs; it cannot establish failure detection performance. The high validation F1 (GRU 0.991) is a model-selection result on one drive, not an independent final result. Device, collection year and route shift are confounded. Future evaluation needs multiple complete independent drives and repeated-route/device holdouts.

The navigation simulation contains 15 generated trips. Its observation geometry, biases, signal statistics and acceleration are simplified. Real WLS computation on simulated pseudoranges does not make the experiment real data. In this simulation fixed filtering outperformed adaptive filtering; in the 11-epoch real smoke test adaptive filtering also did not improve WLS RMSE. No consistent learned-filter improvement has been established.

Signal statistics and model risk alone cannot establish an NLOS cause. Diagnostics must distinguish recorded facts from hypotheses and missing evidence. The optional LLM adapters have protocol tests; no live API quality evaluation is claimed without configured credentials.

## Reproducibility

Reports record dataset SHA-256, seed, features/coverage, exact trip memberships, validation thresholds, history and metrics. Public source URLs are commit-pinned with size and SHA-256 in `config/public_sources.json`; preprocessing records alignment/drop/fallback counts. `requirements-lock.txt` is the tested Python 3.12 dependency snapshot excluding platform-specific PyTorch; CPU PyTorch version is installed separately as documented. This is not a cross-platform hash lock.

Published result summaries and limitations are in `docs/experiments/v0_2.md`. Third-party raw data and model binaries are excluded from Git. Load only model files you generated or trust, since joblib is a pickle-based format.
