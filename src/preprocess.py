import pandas as pd
import numpy as np

# Column names
INDEX_COLS = ['unit_nr', 'time_cycles']
SETTING_COLS = ['setting_1', 'setting_2', 'setting_3']
SENSOR_COLS = [f's_{i}' for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Sensors with near-zero variance in FD001 (constant, carry no info)
DROP_SENSORS = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']

# Clip RUL at 125 — engines don't degrade linearly from cycle 1,
# so we cap early-life RUL to avoid the model overfitting to "healthy" noise
RUL_CLIP = 125


def load_data(data_dir='data'):
    train = pd.read_csv(f'{data_dir}/train_FD001.txt', sep=r'\s+', header=None, names=ALL_COLS)
    test  = pd.read_csv(f'{data_dir}/test_FD001.txt',  sep=r'\s+', header=None, names=ALL_COLS)
    rul   = pd.read_csv(f'{data_dir}/RUL_FD001.txt',   sep=r'\s+', header=None, names=['RUL'])
    return train, test, rul


def add_rul(df):
    """Add RUL column to training data (max_cycle - current_cycle per engine)."""
    max_cycles = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_nr', 'max_cycle']
    df = df.merge(max_cycles, on='unit_nr')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df['RUL'] = df['RUL'].clip(upper=RUL_CLIP)
    df.drop(columns=['max_cycle'], inplace=True)
    return df


def add_test_rul(test, rul_df):
    """Add ground-truth RUL to test set (take last cycle per engine + known RUL)."""
    max_cycles = test.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_nr', 'max_cycle']
    max_cycles['true_rul'] = rul_df['RUL'].values
    test = test.merge(max_cycles, on='unit_nr')
    test['RUL'] = test['max_cycle'] - test['time_cycles'] + test['true_rul']
    test['RUL'] = test['RUL'].clip(upper=RUL_CLIP)
    test.drop(columns=['max_cycle', 'true_rul'], inplace=True)
    return test


def add_rolling_features(df, window=30):
    """Rolling mean and std for each sensor — captures degradation trend."""
    df = df.sort_values(['unit_nr', 'time_cycles']).reset_index(drop=True)
    active_sensors = [s for s in SENSOR_COLS if s not in DROP_SENSORS]
    for sensor in active_sensors:
        grp = df.groupby('unit_nr')[sensor]
        df[f'{sensor}_roll_mean'] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'{sensor}_roll_std']  = grp.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    return df


def drop_useless_sensors(df):
    return df.drop(columns=DROP_SENSORS, errors='ignore')


def get_feature_cols(df):
    exclude = ['unit_nr', 'time_cycles', 'RUL']
    return [c for c in df.columns if c not in exclude]


def preprocess(data_dir='data'):
    train, test, rul = load_data(data_dir)

    train = add_rul(train)
    test  = add_test_rul(test, rul)

    train = drop_useless_sensors(train)
    test  = drop_useless_sensors(test)

    train = add_rolling_features(train)
    test  = add_rolling_features(test)

    feature_cols = get_feature_cols(train)

    # For test set, take only the LAST cycle per engine (that's what we're predicting RUL for)
    test_last = test.groupby('unit_nr').last().reset_index()

    X_train = train[feature_cols]
    y_train = train['RUL']
    X_test  = test_last[feature_cols]
    y_test  = test_last['RUL']

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"RUL range (train): {y_train.min():.0f} – {y_train.max():.0f}")
    print(f"RUL range (test):  {y_test.min():.0f}  – {y_test.max():.0f}")

    return X_train, y_train, X_test, y_test, feature_cols, train


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, feature_cols, train_df = preprocess()
