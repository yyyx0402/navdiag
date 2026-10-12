"""Report acquisition coverage separately from conditional positioning accuracy."""
import numpy as np
import pandas as pd

from navdiag.fusion import position_metrics


def availability_summary(events: pd.DataFrame) -> dict:
    required = {'timestamp_ms', 'solution_available', 'reference_time_matched',
                'reference_quality_allowed', 'scorable'}
    if events.empty or not required <= set(events):
        raise ValueError('Availability requires a nonempty acquisition event table')
    ordered = events.sort_values('timestamp_ms')
    times = ordered.timestamp_ms.to_numpy(float) / 1000
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Availability timestamps must be unique and finite')
    for column in required - {'timestamp_ms'}:
        if not ordered[column].isin([True, False]).all():
            raise ValueError('Availability status columns must be boolean')
    available = ordered.solution_available.to_numpy(bool)
    eligible = (ordered.reference_time_matched & ordered.reference_quality_allowed).to_numpy(bool)
    scored = ordered.scorable.to_numpy(bool)
    if not np.array_equal(scored, available & eligible):
        raise ValueError('Scorable epochs must equal available fixes with eligible references')
    cadence = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    longest = 0.0
    start = None
    for index, valid in enumerate(available):
        if not valid and start is None:
            start = index
        if start is not None and (valid or index == len(times) - 1):
            end = index - 1 if valid else index
            longest = max(longest, float(times[end] - times[start] + cadence))
            start = None
    success_times = times[available]
    return {
        'raw_epochs': len(times), 'raw_time_span_s': float(times[-1] - times[0]),
        'nominal_cadence_s': cadence, 'solution_epochs': int(available.sum()),
        'solution_availability': float(available.mean()),
        'reference_time_matched_epochs': int(ordered.reference_time_matched.sum()),
        'eligible_reference_epochs': int(eligible.sum()), 'scored_epochs': int(scored.sum()),
        'scored_fraction_of_raw': float(scored.mean()),
        'solution_availability_on_eligible_reference': float(scored.sum() / eligible.sum()) if eligible.any() else None,
        'longest_solution_outage_s': longest,
        'outage_duration_definition': 'Last minus first missing epoch plus median raw epoch cadence.',
        'causal_filter_segments': int(1 + (np.diff(success_times) > 10).sum()) if len(success_times) else 0,
        'causal_gru_segments': int(1 + (np.diff(success_times) > 5).sum()) if len(success_times) else 0,
        'filter_reset_gap_s': 10, 'gru_reset_gap_s': 5,
    }


def coverage_position_metrics(errors: np.ndarray, eligible_reference_epochs: int) -> dict:
    result = position_metrics(errors)
    if eligible_reference_epochs < result['input_epochs'] or eligible_reference_epochs < 1:
        raise ValueError('Eligible reference denominator must cover scored epochs')
    values = np.asarray(errors, float)
    result['eligible_reference_epochs'] = eligible_reference_epochs
    result['available_and_within_error_limits_m'] = {
        str(limit): float((np.isfinite(values) & (values <= limit)).sum() / eligible_reference_epochs)
        for limit in [1, 2, 5, 10]
    }
    return result
