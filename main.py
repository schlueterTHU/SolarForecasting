import data_preprocessing.preprocessing as pp
from data_preprocessing.WindowGenerator import WindowGenerator
import models.train_model as tm

import config as cfg
import config_models as cfg_mod
import visualization.plotting as plotting

# import numpy as np
import pandas as pd
import pickle


def main():
    # models = prepare_models(cfg_mod.models)
    # print_metrics(models)
    plotting.multi_plot(cfg_mod.cities, cfg.errors['metric'])
    ################ Plot RMSE ###################
    # plotting.plot_metric(models, 'skewness').show()
    # plotting.plot_metric(models, 'rmse').show()
    # plotting.plot_metric(models, 'mae').show()
    # plotting.multi_plot_days(models, cfg.plotting['days'])
    # for day in cfg.plotting['days']:
    #     plotting.plot_forecast(models, day.year, day.month, day.day, day.hour).show()


def prepare_models(models):
    for model in models:
        df = pd.read_pickle('data/pickles/' + model['city'] + '.pickle')
        train, val, test, num_features, date_time, column_indices = \
            pp.preprocess(df, model['fields'], city=model['city'], time=True)
        model['train'] = train
        model['val'] = val
        model['test'] = test
        model['test_datetime'] = date_time
        model['num_features'] = num_features
        model['column_indices'] = column_indices

        with open('./saver/outputs/scaler/output_scaler_' + model['city'] + '.pckl', 'rb') as file_scaler:
            model['scaler'] = pickle.load(file_scaler)

        model['window'] = WindowGenerator(input_width=cfg.prediction['input_len'],
                                          label_width=cfg.prediction['num_predictions'],
                                          train_df=model['train'], val_df=model['val'], test_df=model['test'],
                                          shift=cfg.prediction['num_predictions'],
                                          label_columns=[cfg.label],
                                          )
        model['model'] = tm.build_model(tm.choose_model(model), model['window'],
                                        './checkpoints/' + model['city'] + '/' + model['type'] + '_' +
                                        model['city'] + model.get('number', ''),
                                        train=model['train_bool'])
        if model.get('baseline'):
            model['baseline']['model'] = tm.build_model(tm.choose_model(model['baseline']), model['window'],
                                                        './checkpoints/' + model['city'] + '/' + model['type'] + '_' +
                                                        model['city'] + model.get('number', ''),
                                                        train=model['train_bool'])
            model['baseline']['rmse'], model['baseline']['mae'], model['baseline']['skewness'] = \
                model['window'].get_metrics(model['baseline']['model'], model['scaler'], model['city'])
        model['rmse'], model['mae'], model['skewness'] = \
            model['window'].get_metrics(model['model'], model['scaler'], model['city'])
    return models


def print_metrics(models):
    rmse_baseline = None
    mae_baseline = None
    for model in models:
        if model.get('baseline'):
            rmse_baseline = sum(model['baseline']['rmse']) / 24
            mae_baseline = sum(model['baseline']['mae']) / 24
    for model in models:
        rmse_model = sum(model['rmse'])/24
        mae_model = sum(model['mae'])/24
        print(model['name'] + ': ' + model['city'].capitalize() + ' ' + model['type'])
        if rmse_baseline is not None:
            print('Percentage RMSE: ', rmse_model/rmse_baseline)
            print('Percentage MAE: ', mae_model/mae_baseline)
        else:
            print('No Baseline defined, can not give RMSE Percentage!')
        print('Total RMSE: ', rmse_model)
        print('Total MAE: ', mae_model)


if __name__ == '__main__':
    main()
