# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
import models.train_model as tm
import pandas as pd
import config as cfg
import config_models as cfg_mod
import data_preprocessing.preprocessing as pp
import pickle
from visualization.plotting import plot_forecast
from data_preprocessing.WindowGenerator import WindowGenerator


def main():
    models = cfg_mod.models
    for model in models:
        df = pd.read_pickle('data/pickles/' + model['city'] + '.pickle')
        train, val, test, num_features, date_time, column_indices = \
            pp.preprocess(df, model['fields'], city=model['city'], time=True)
        print(df.head())


def main1():
    models = cfg_mod.models
    for model in models:
        df = pd.read_pickle('data/pickles/' + model['city'] + '.pickle')
        train, val, test, num_features, date_time, column_indices = \
            pp.preprocess(df, model['fields'], city=model['city'], time=True)
        model['train'] = train
        model['val'] = val
        model['test'] = test
        model['num_features'] = num_features
        model['test_datetime'] = date_time
        model['column_indices'] = column_indices

        model['window'] = WindowGenerator(input_width=cfg.prediction['input_len'],
                                          label_width=cfg.prediction['num_predictions'],
                                          train_df=model['train'], val_df=model['val'], test_df=model['test'],
                                          shift=cfg.prediction['num_predictions'],
                                          label_columns=[cfg.label])

        with open('./saver/outputs/scaler/output_scaler_' + model['city'] + '.pckl', 'rb') as file_scaler:
            model['scaler'] = pickle.load(file_scaler)

        model['model'] = tm.build_model(tm.choose_model(model), model['window'],
                                        './checkpoints/' + model['city'] + '/' + model['type'] + '_' +
                                        model['city'] + model.get('number', ''),
                                        train=model['train_bool'])
        model['model'].summary()
    plot_forecast(models, 2020, 3, 12, 20)
    # plot_forecast(models, 2020, 8, 31, 0)
    # plot_forecast(models, 2020, 7, 20, 4)
    # plot_forecast(models, 2020, 6, 15, 0)
    print('done')


if __name__ == '__main__':
    main()
