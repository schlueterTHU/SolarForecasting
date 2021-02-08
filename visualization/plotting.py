import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter

import models.train_model as tm
import config as cfg
# import pickle


def get_day_data(data, date_time, year, month, day, hour):
    start = date_time.index(pd.Timestamp(year, month, day, hour))
    mid = start+cfg.prediction['input_len']
    end = mid+cfg.prediction['num_predictions']
    # return day_date_time, input, label
    return date_time[start:end], data[start:mid].to_numpy(), data[mid:end][cfg.label].to_numpy()


def plot_forecast(data, year, month, day, hour):
    day_date_time, inputs, label = get_day_data(data[0]['test'], data[0]['test_datetime'], year, month, day, hour)
    for model in data:
        pred = tm.get_predictions(model['model'],
                                  inputs.reshape(1, cfg.prediction['num_predictions'], model['num_features']),
                                  label.reshape(1, cfg.prediction['num_predictions']),
                                  model['scaler']).reshape(-1)
        # plt.scatter(day_date_time[24:], pred,
        #             marker='X', edgecolors='k', label=model['type'].upper(), s=64)
        #             # c='#ff7f0e', s=64)
        plt.plot(day_date_time[cfg.prediction['num_predictions']:], pred, label=model['type'],
                 color=model.get('plotting', {}).get('color', None),
                 marker=model.get('plotting', {}).get('marker', 'x'),
                 linestyle=model.get('plotting', {}).get('linestyle', '-'),
                 linewidth=2, markersize=5, zorder=2.7)
    label = data[0]['scaler'].inverse_transform(label.reshape(-1, 1))
    inputs = data[0]['scaler'].inverse_transform(inputs[:, 0].reshape(-1, 1))
    plt.plot(day_date_time, np.concatenate([label.reshape(-1), inputs[:, 0]]), color='k',
             label='True', marker='.', markersize=3, zorder=2.6)
    # plt.plot(day_date_time[:24], inputs[:, 0], color='k',
    #          label='Inputs', marker='.', zorder=-10)
    # plt.plot(day_date_time[24:], label, color='k', marker='.', zorder=-10)
    # plt.scatter(day_date_time[24:], label,
    #             edgecolors='k', label='Labels', c='#2ca02c', s=64)
    # plt.figure(figsize=(12, 8))
    plt.title(data[0]['name'] + ' ' + str(day_date_time[0].date()))
    plt.ylabel(f'global irradiation [$Wh/m^2$]')
    # plt.xlabel('time [h]')
    plt.xticks(day_date_time[::6], map(pd.Timestamp.time, day_date_time[::6]), rotation=20)
    formatter = DateFormatter('%H:%M')
    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_rmse(models):
    for model in models:
        plt.plot(range(24), model['rmse'], label=model['name'],
                 color=model.get('plotting', {}).get('color', None),
                 marker=model.get('plotting', {}).get('marker', 'x'),
                 linestyle=model.get('plotting', {}).get('linestyle', '-'),
                 linewidth=2, markersize=8)
        if model.get('baseline'):
            plt.plot(range(24), model['baseline']['rmse'], label=model['name']+'_Baseline',
                     color=model.get('baseline').get('plotting', {}).get('color', None),
                     marker=model.get('baseline').get('plotting', {}).get('marker', 'x'),
                     linestyle=model.get('baseline').get('plotting', {}).get('linestyle', '-'),
                     linewidth=2, markersize=8)
    plt.legend()
    plt.title('Root Mean Squared Error with N=%d samples' % (cfg.data['test_data_size']))
    plt.xlabel('Time [h]')
    plt.ylabel('standard deviation (RMSE) in W/m^2')
    plt.grid(True)
    plt.show()
