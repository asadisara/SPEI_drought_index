import numpy as np

def find_extrema(series, n_abs=2):
    diff = series.diff()
    max_candidates = series[(diff.shift(-1) < 0) & (diff > 0)]
    min_candidates = series[(diff.shift(-1) > 0) & (diff < 0)]

    real_max = max_candidates[max_candidates > 0].idxmax() if not max_candidates[max_candidates > 0].empty else None
    real_min = min_candidates[min_candidates < 0].idxmin() if not min_candidates[min_candidates < 0].empty else None

    max_abs_indices = sorted(max_candidates.index, key=lambda x: abs(series.loc[x]), reverse=True)[:n_abs]
    min_abs_indices = sorted(min_candidates.index, key=lambda x: abs(series.loc[x]), reverse=True)[:n_abs]

    return {
        "real_max": real_max,
        "real_min": real_min,
        "top_max_abs": max_abs_indices,
        "top_min_abs": min_abs_indices
    }


def find_midpoints(series, start_date):
    values = series.values
    index = series.index
    mid_candidates = []

    i = 0
    while i < len(values) - 2:
        if values[i] < values[i + 1]:
            start = i
            while i < len(values) - 1 and values[i] < values[i + 1]:
                i += 1
            end = i
        elif values[i] > values[i + 1]:
            start = i
            while i < len(values) - 1 and values[i] > values[i + 1]:
                i += 1
            end = i
        else:
            i += 1
            continue

        if end > start + 1:
            start_val = values[start]
            end_val = values[end]
            mid_val = (start_val + end_val) / 2

            segment = values[start + 1:end]
            segment_idx = index[start + 1:end]
            diffs = np.abs(segment - mid_val)
            best_j = np.argmin(diffs)
            mid_index = segment_idx[best_j]
            diff = abs(end_val - start_val)
            mid_candidates.append((mid_index, diff))

    mid_candidates = [(idx, diff) for idx, diff in mid_candidates if idx >= start_date]
    mid_candidates_sorted = sorted(mid_candidates, key=lambda x: x[1], reverse=True)[:3]

    return [idx for idx, _ in mid_candidates_sorted]
