import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
# from pmdarima.arima import auto_arima
import numpy as np
from sklearn.metrics import mean_squared_error
from warnings import catch_warnings, filterwarnings
import random
import pickle

import config as cfg


def create_sarima_model(df):
    mod = sm.tsa.statespace.SARIMAX(df,
                                    order=(0, 1, 1),
                                    seasonal_order=(0, 1, 1, 24),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    # print(results.summary().tables[1])
    return results


def eval_sarima(data):
    mod = sm.tsa.statespace.SARIMAX(data,
                                    order=(1, 0, 1),
                                    exog=None,
                                    seasonal_order=(1, 0, 0, 24),
                                    trend=None,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    with catch_warnings():
        filterwarnings("ignore")
        trained_model = mod.fit(disp=0)
    return trained_model.forecast(steps=cfg.prediction['num_predictions'])


def get_data(i, data):
    train = data[i:i+cfg.sarima['train_len']]
    label = data[i+cfg.sarima['train_len']:i+cfg.sarima['train_len']+cfg.prediction['num_predictions']]
    return train, label


def eval_rmse(df):
    n = len(df)
    ls = list(range(n-(cfg.sarima['train_len']+cfg.prediction['num_predictions'])))
    random.shuffle(ls)
    nos = min(1200, len(ls))
    pred = []
    label = []
    for count, i in enumerate(ls[:nos]):
        train, labels = get_data(i, df)
        forecast = eval_sarima(train)
        pred.append(forecast)
        label.append(labels)
        print(count)
    pred = np.array(pred)
    label = np.array(label)
    mse = mean_squared_error(label, pred, multioutput='raw_values')
    rmse = np.sqrt(mse.transpose())
    return rmse


def main():
    df = pd.read_pickle('../data/pickles/data_pickle.pickle')
    rmse = eval_rmse(np.array(df[[cfg.label]]).reshape(-1))
    sarima_rmse = open('../saver/outputs/rmse/sarima101.pckl', 'wb')
    pickle.dump(rmse, sarima_rmse)
    plt.plot(range(24), rmse, 'x-', label='SARIMA', linewidth=2, markersize=8)
    plt.show()


if __name__ == '__main__':
    main()
