"""
added features:

trend mask
EWMA for velocity & acceleration
sin & cos time
open raw log1p normalized
diff features for OHLC
percent diff for volume

"""
from preprocess import _mask_trends, _Velocity
import hyperparams as hp
import pandas as pd
import numpy as np
import os

FILENAME = 'training_data.csv'

"""
classifier stats : vel_alpha, accel_alpha, vel_ratio, trend_gate
"""
def preprocess(filename, classifier_stats = (0.06, 0.3, 0.012, 0.12)):

    df = pd.read_csv(filename)

    # classify points as trendy / non trendy
    trend_mask, _ = _mask_trends(df, *classifier_stats)
    print(f"Trendy points: {trend_mask.sum() / len(trend_mask)}")
    df['is_mask'] = trend_mask

    # time features
    def normalize_time(row):
        h, m, s = map(int, row[11:].split(':'))
        seconds = h * 3600 + m * 60 + s
        radians = 2 * np.pi * seconds / 86400
        return np.sin(radians), np.cos(radians) 

    df['sin_time'], df['cos_time'] = zip(*df['datetime'].apply(normalize_time))

    df['volume_diff'] = df['volume'].pct_change()
    df.iloc[0, df.columns.get_loc('volume_diff')] = 0
        
    # raw diff features for OHLC & normalize raw values
    for feature in ('open', 'high', 'low', 'close'):
        df[f'{feature}_diff'] = df[feature].diff()
        df.iloc[0, df.columns.get_loc(f'{feature}_diff')] = 0

    for feature in ('high_diff', 'low_diff', 'close_diff'):
        df[f'{feature}_asinh'] = np.asinh(df[feature])
        df.iloc[0, df.columns.get_loc(f'{feature}_asinh')] = 0

    for feature in ('open', 'volume'):
        df[f'{feature}_r'] = np.log1p(df[feature])

    # get single and double EWMA for velocity and acceleration
    volatility = _Velocity(hp.VEL_ALPHA, hp.ACCEL_ALPHA)    
    def calc_velocity(volatility, value):
        volatility.add(value)
        return volatility.value, volatility.acceleration.value
    
    df['velocity'], df['acceleration'] = zip(*df['close_diff'].apply(lambda x: calc_velocity(volatility, x)))
    
    # drop unnecessary columns
    df = df.drop(columns=['active_contract', 'contract_type', 'adjustment_applied', 'datetime'])
    df.to_csv("training_data_p.csv", index=False)

if __name__ == "__main__":
    if not FILENAME.replace('.csv', '_p.csv') in os.listdir(os.getcwd()):
        preprocess('training_data.csv')