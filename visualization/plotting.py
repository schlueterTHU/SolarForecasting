import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter

import models.train_model as tm
import config as cfg
import config_models as cfg_mod
from main import prepare_models
# import pickle


def get_day_data(data, date_time, year, month, day, hour):
    # start = date_time.index(pd.Timestamp(year, month, day, hour))
    # mid = start+cfg.prediction['input_len']
    # end = mid+cfg.prediction['num_predictions']
    mid = date_time.index(pd.Timestamp(year, month, day, hour))
    start = mid-cfg.prediction['input_len']
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
        plt.plot(day_date_time[cfg.prediction['num_predictions']:], pred, label=model['name'],
                 color=model.get('plotting', {}).get('color', None),
                 marker=model.get('plotting', {}).get('marker', 'x'),
                 linestyle=model.get('plotting', {}).get('linestyle', '-'),
                 linewidth=2, markersize=5, zorder=2.7)
    label = data[0]['scaler'].inverse_transform(label.reshape(-1, 1))
    inputs = data[0]['scaler'].inverse_transform(inputs[:, 0].reshape(-1, 1))
    plt.plot(day_date_time, np.concatenate([inputs[:, 0], label.reshape(-1)]), color='k',
             label='True', marker='.', markersize=3, zorder=2.6)
    # plt.title(data[0]['city'].capitalize() + ' ' + str(day) + '.' + str(month) + '.' + str(year))
    plt.title(data[0]['city'].capitalize() + ', ' + day_date_time[24].date().strftime('%d.%m.%y'))
    plt.ylabel(f'global irradiation [$Wh/m^2$]')
    plt.xticks(day_date_time[::6], map(pd.Timestamp.time, day_date_time[::6]), rotation=20)
    formatter = DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.grid(True)
    return plt


def multi_plot_days(models, days):
    plt.figure(figsize=(15, 4.5))
    for i, day in enumerate(days):
        plt.subplot(int((len(days)+1)/2), 2, i + 1)
        plot_forecast(models, day.year, day.month, day.day, day.hour)
    plt.tight_layout()
    plt.show()


def plot_metric(models, metric):
    for model in models:
        plt.plot(range(1, 25), model[metric], label=model['name'],
                 color=model.get('plotting', {}).get('color', None),
                 marker=model.get('plotting', {}).get('marker', 'x'),
                 linestyle=model.get('plotting', {}).get('linestyle', '-'),
                 linewidth=2, markersize=6)
        if model.get('baseline'):
            plt.plot(range(1, 25), model['baseline'][metric], label='SN',
                     color=model.get('baseline').get('plotting', {}).get('color', None),
                     marker=model.get('baseline').get('plotting', {}).get('marker', 'x'),
                     linestyle=model.get('baseline').get('plotting', {}).get('linestyle', '-'),
                     linewidth=2, markersize=6)
    plt.legend(prop={'size': 10}, loc='upper right')
    # plt.title('Mean Absolute Error with N=%d samples' % (cfg.data['test_data_size']))
    plt.title(models[0]['city'].capitalize())
    # plt.xticks(range(26)[::5])
    plt.xticks([1, 5, 10, 15, 20, 24])
    plt.xlabel('Time [h]')
    plt.legend(prop={'size': 10})
    if metric != 'skewness':
        plt.ylabel(metric.upper() + ' [$Wh/m^2$]')
    else:
        plt.ylabel(metric.capitalize())
        # plt.legend(prop={'size': 10}, loc='upper right')
    plt.grid(True)
    # plt.show()
    return plt


def multi_plot(cities, metric):
    plt.figure(figsize=(15, 9))
    for i, city in enumerate(cities):
        models = prepare_models(cfg_mod.return_models(city))
        plt.subplot(int((len(cities)+1)/2), 2, i + 1)
        # choose_metric(metric, models)
        plot_metric(models, metric)
        # plt.plot(choose_metric(metric, models))
    plt.tight_layout()
    plt.show()


# def plot_rmse(models):
#     for model in models:
#         plt.plot(range(24), model['rmse'], label=model['name'],
#                  color=model.get('plotting', {}).get('color', None),
#                  marker=model.get('plotting', {}).get('marker', 'x'),
#                  linestyle=model.get('plotting', {}).get('linestyle', '-'),
#                  linewidth=2, markersize=6)
#         if model.get('baseline'):
#             plt.plot(range(24), model['baseline']['rmse'], label='SN',
#                      color=model.get('baseline').get('plotting', {}).get('color', None),
#                      marker=model.get('baseline').get('plotting', {}).get('marker', 'x'),
#                      linestyle=model.get('baseline').get('plotting', {}).get('linestyle', '-'),
#                      linewidth=2, markersize=6)
#     plt.legend(prop={'size': 10})
#     # plt.title(models[0]['city'].capitalize() +
#     # ':  Root Mean Squared Error with N=%d samples' % (cfg.data['test_data_size']))
#     plt.title(models[0]['city'].capitalize())
#     plt.xlabel('Time [h]')
#     plt.ylabel('RMSE in W/m^2')
#     plt.grid(True)
#     return plt
#     # plt.show()
#
#
# def plot_mae(models):
#     for model in models:
#         plt.plot(range(24), model['mae'], label=model['name'],
#                  color=model.get('plotting', {}).get('color', None),
#                  marker=model.get('plotting', {}).get('marker', 'x'),
#                  linestyle=model.get('plotting', {}).get('linestyle', '-'),
#                  linewidth=2, markersize=6)
#         if model.get('baseline'):
#             plt.plot(range(24), model['baseline']['mae'], label='SN',
#                      color=model.get('baseline').get('plotting', {}).get('color', None),
#                      marker=model.get('baseline').get('plotting', {}).get('marker', 'x'),
#                      linestyle=model.get('baseline').get('plotting', {}).get('linestyle', '-'),
#                      linewidth=2, markersize=6)
#     plt.legend(prop={'size': 10})
#     # plt.title('Mean Absolute Error with N=%d samples' % (cfg.data['test_data_size']))
#     plt.title(models[0]['city'].capitalize())
#     plt.xlabel('Time [h]')
#     plt.ylabel('MAE in W/m^2')
#     plt.grid(True)
#     # plt.show()
#     return plt


# def choose_metric(metric, models):
#     metric_dict = {
#         'RMSE': plot_rmse,
#         'MAE': plot_mae,
#     }
#     func = metric_dict.get(metric, lambda x: "Invalid metric")
#     return func(models)
