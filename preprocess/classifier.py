from typing import TypedDict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABEL_SIZE = 5
WINDOW_SIZE = 120


class _EWMA:
    """Exponentially Weighted Moving Average tracker."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None
        self.sign = ''

    def add(self, value: float):
        if self.value is None:
            self.value = value
            self.sign = _signcheck(value)
        else:
            self.value = self.alpha * value + (1 - self.alpha) * self.value

    def reset(self):
        self.value = None


class _Velocity(_EWMA):
    """EWMA velocity tracker with derived acceleration (second-order EWMA)."""

    def __init__(self, vel_alpha, accel_alpha):
        super().__init__(vel_alpha)
        self.previous_value = None
        self.acceleration = _EWMA(accel_alpha)

    def add(self, value: float):
        self.previous_value = self.value
        super().add(value)

        if self.previous_value is not None:
            accel = self.value - self.previous_value
            self.acceleration.add(accel)
        else:
            self.acceleration.add(0)

    def reset(self):
        super().reset()
        self.previous_value = None
        self.acceleration.reset()


def _signcheck(num: float):
    """Return the sign of a number as a character: '+', '-', or '0'."""
    if num is None or num == 0:
        return '0'
    return '+' if num > 0 else '-'


def _mask_trends(df: pd.DataFrame, vel_alpha=0.06, accel_alpha=0.3,
                 vel_ratio=0.012, trend_gate=0.12):
    """Identify trending segments in price data using EWMA velocity/acceleration heuristics.

    Returns a binary mask (1 = trending) and the count of detected trends.
    """

    class _TrendMarker(TypedDict):
        start: int
        stop: Optional[int]
        size: int

    df_close = df[['close']].copy()

    # Non-overlapping windows of size LABEL_SIZE that average velocities
    labels = (df_close - df_close.shift(1)).rolling(window=LABEL_SIZE).mean()[LABEL_SIZE - 1::LABEL_SIZE].reset_index(drop=True)

    # Cumulative movement over WINDOW_SIZE / LABEL_SIZE label windows
    num_label_windows = int(WINDOW_SIZE / LABEL_SIZE)
    large_windows = labels['close'].abs().rolling(window=num_label_windows).sum()[num_label_windows - 1:].to_frame().reset_index(drop=True)

    trending = False
    current_sign = ''

    trend_tracker = _Velocity(vel_alpha, accel_alpha)
    trend_mask = pd.Series(0, index=df.index)
    valid_trends = 0

    for index in range(len(large_windows) - 1):
        if not trending:
            window_index = index

        diff_index = index + num_label_windows
        window = large_windows.iloc[window_index, 0]
        label = labels.iloc[diff_index, 0]

        trend_tracker.add(label)
        velocity = trend_tracker.value
        acceleration = trend_tracker.acceleration.value

        if not trending and (abs(velocity) / abs(window) > vel_ratio or _signcheck(velocity) != _signcheck(acceleration)):
            trending = True
            current_sign = _signcheck(velocity)
            possible_trend = _TrendMarker(start=diff_index, stop=None, size=1)

        if trending:
            possible_trend["size"] += 1

            if possible_trend["size"] == 3:
                # Confirm trend: acceleration must align with velocity and exceed threshold
                if _signcheck(acceleration) != current_sign or abs(acceleration) < trend_gate * abs(velocity):
                    trending = False
                    del possible_trend

            elif possible_trend["size"] > 3:
                # End trend when velocity and acceleration both flip sign
                if _signcheck(velocity) != current_sign and _signcheck(acceleration) != current_sign:
                    possible_trend["stop"] = diff_index
                    valid_trends += 1
                    trend_mask[LABEL_SIZE * possible_trend["start"]:LABEL_SIZE * (possible_trend["stop"] + 1)] = 1
                    trending = False
                    del possible_trend

    return trend_mask, valid_trends


def graph_trends(df: pd.DataFrame, trend_indices: pd.Series):
    """Visualize detected trends overlaid on the close price series."""
    plt.figure(figsize=(14 * int(len(df) / 1500), 6))

    x = df.index
    y = df['close'].values

    plt.plot(x, y, color='blue', linewidth=1.5, label='Non-Trend')

    trend_mask_indices = np.where(trend_indices == 1)[0]
    if trend_mask_indices.size > 0:
        breaks = np.where(np.diff(trend_mask_indices) > 1)[0]
        segment_starts = np.insert(breaks + 1, 0, 0)
        segment_ends = np.append(breaks, trend_mask_indices.size - 1)

        for start, end in zip(segment_starts, segment_ends):
            seg = trend_mask_indices[start:end + 1]
            plt.plot(x[seg], y[seg], color='red', linewidth=2.2, label='Trend' if start == 0 else "")

    plt.title('Highlighted Trends vs. Non-Trends', fontsize=14)
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig("trends.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv('training_data.csv')
    trend_mask, trends = _mask_trends(df)
