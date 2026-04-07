import os

import numpy as np
import pandas as pd

from preprocess import _mask_trends, _Velocity
import hyperparams as hp

FILENAME = 'training_data.csv'


def preprocess(filename, classifier_stats=(0.06, 0.3, 0.012, 0.12)):
    """Preprocess raw OHLCV data into features for the transformer model.

    Adds: trend mask, EWMA velocity/acceleration, sin/cos time encoding,
    OHLC diffs, asinh-normalized diffs, log1p-normalized raw values, volume pct change.
    """
    df = pd.read_csv(filename)

    # Classify points as trending / non-trending
    trend_mask, _ = _mask_trends(df, *classifier_stats)
    print(f"Trendy points: {trend_mask.sum() / len(trend_mask):.4f}")
    df['is_mask'] = trend_mask

    # Cyclical time encoding
    def encode_time(row):
        h, m, s = map(int, row[11:].split(':'))
        seconds = h * 3600 + m * 60 + s
        radians = 2 * np.pi * seconds / 86400
        return np.sin(radians), np.cos(radians)

    df['sin_time'], df['cos_time'] = zip(*df['datetime'].apply(encode_time))

    # Volume percent change
    df['volume_diff'] = df['volume'].pct_change()
    df.iloc[0, df.columns.get_loc('volume_diff')] = 0

    # Raw diff features for OHLC
    for feature in ('open', 'high', 'low', 'close'):
        df[f'{feature}_diff'] = df[feature].diff()
        df.iloc[0, df.columns.get_loc(f'{feature}_diff')] = 0

    # Asinh normalization for high/low/close diffs
    for feature in ('high_diff', 'low_diff', 'close_diff'):
        df[f'{feature}_asinh'] = np.asinh(df[feature])
        df.iloc[0, df.columns.get_loc(f'{feature}_asinh')] = 0

    # Log1p normalization for raw open and volume
    for feature in ('open', 'volume'):
        df[f'{feature}_r'] = np.log1p(df[feature])

    # EWMA velocity and acceleration
    velocity_tracker = _Velocity(hp.VEL_ALPHA, hp.ACCEL_ALPHA)

    def calc_velocity(value):
        velocity_tracker.add(value)
        return velocity_tracker.value, velocity_tracker.acceleration.value

    df['velocity'], df['acceleration'] = zip(*df['close_diff'].apply(calc_velocity))

    # Drop unnecessary columns
    df = df.drop(columns=['active_contract', 'contract_type', 'adjustment_applied', 'datetime'])
    df.to_csv("training_data_p.csv", index=False)


if __name__ == "__main__":
    if FILENAME.replace('.csv', '_p.csv') not in os.listdir(os.getcwd()):
        preprocess(FILENAME)
