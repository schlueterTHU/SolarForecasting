import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

import config as cfg


def preprocess(df, columns, city='', time=True):
    date_time = None
    if time:
        df, date_time = change_date(df)
    df = scale_data(df, './saver/outputs/scaler/output_scaler_' + city + '.pckl', columns)
    column_indices = {name: i for i, name in enumerate(df.columns)}
    num_features = df.shape[1]
    train_df, val_df, test_df, test_dt = split_data(df, date_time)
    return train_df, val_df, test_df, num_features, test_dt, column_indices


def change_date(df):
    date_time = pd.to_datetime(df.pop('date'), format='%d.%m.%Y %H:%M:%S')
    timestamp_s = date_time.map(datetime.datetime.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    return df, list(date_time)


def split_data(df, date_time):
    n = len(df) - cfg.data['test_data_size']
    train_df = df[0:int(n * cfg.data['train_data_perc'])]
    val_df = df[int(n * cfg.data['train_data_perc']):n]
    test_df = df[-cfg.data['test_data_size']:]
    return train_df, val_df, test_df, date_time[-cfg.data['test_data_size']:]


def scale_data(df, filepath, columns):
    output_scaler = MinMaxScaler(feature_range=(0, 1))
    output_scaler.fit(df[[cfg.label]])
    file_scaler = open(filepath, 'wb')
    pickle.dump(output_scaler, file_scaler)

    scaler_all = MinMaxScaler(feature_range=(0, 1))
    df = pd.DataFrame(scaler_all.fit_transform(df[columns]), columns=columns)
    return df
