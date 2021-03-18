import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import skew
import pickle

import models.train_model as tm

import config as cfg


def determine_weight(inputs):
    weights = np.zeros(inputs.shape)
    weights[inputs != 0] = 1
    return weights


class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self._example = None

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        # if self.label_columns is not None:
        #     labels = tf.stack(
        #         [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        #         axis=-1)
        labels = labels[:, :, self.column_indices[self.label_columns[0]]]

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width])

        return inputs, labels

    def plot(self, model=None, plot_col=cfg.label, max_subplots=3, myscaler=None):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        if myscaler is None:
            file_scaler = open('saver/outputs/scaler/scaler.pckl', 'rb')
            scaler = pickle.load(file_scaler)
        else:
            scaler = myscaler

        plt.suptitle(model.my_name)
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} in Wh/m^2')
            plt.plot(self.input_indices, scaler.inverse_transform(inputs[n, :, plot_col_index].numpy().reshape(-1, 1)),
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, scaler.inverse_transform(
                labels[n, :, label_col_index].numpy().reshape(-1, 1)),
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = tm.get_predictions(model, inputs, labels, scaler)
                plt.scatter(self.label_indices, predictions[n, :],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()
        plt.xlabel('Time [h]')
        plt.show()

    def get_metrics(self, model, scaler, city='unknown'):
        pred = []
        label = []
        weight = []
        err = []
        for data in self.test:
            inputs, labels = data
            labels = labels.numpy()
            # labels = labels[:, :, 0].numpy()
            predictions = tm.get_predictions(model, inputs.numpy(), labels, scaler)
            pred.extend(predictions)
            label.extend(labels)
            weight.extend(determine_weight(labels))
            err_help = scaler.inverse_transform(labels)-predictions
            err_help[labels == 0] = float('NaN')
            err.extend(err_help)
        pred = np.array(pred)
        label = scaler.inverse_transform(np.array(label))
        # print(model.model_type)
        # err_test = label-pred
        err = np.array(err)
        skewness = skew(err, nan_policy='omit')
        # if model.model_type == 'lstm_conv':
        #     plt.hist(err[:, 5], bins=50)
        #     plt.title('LSTMconv')
        #     plt.xlabel('error')
        #     plt.ylabel('frequency')
        #     plt.show()
        if cfg.errors['error_to_excel']:
            writepath = 'saver/outputs/errors/'+city+'.xlsx'
            mode = 'a' if os.path.exists(writepath) else 'w'
            sheet = model.model_type
            err_df = pd.DataFrame(err)
            err_df.columns = ['hour'+str(i) for i in range(cfg.prediction['num_predictions'])]
            with pd.ExcelWriter(writepath, mode=mode) as writer:
                # if sheet in writer.book.sheetnames:
                #     writer.book.remove(writer.book[sheet])
                err_df.to_excel(writer, sheet, float_format='%.5f')
                writer.save()
        rmse = mean_squared_error(label, pred, sample_weight=weight, multioutput='raw_values', squared=False)
        # rmse = mean_squared_error(label, pred, multioutput='raw_values', squared=False)
        mae = mean_absolute_error(label, pred, sample_weight=weight, multioutput='raw_values')
        # mae = mean_absolute_error(label, pred, multioutput='raw_values')
        return rmse, mae, skewness

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=cfg.data['batch_size'], )

        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.test` dataset
            result = next(iter(self.test))
            # And cache it for next time
            self._example = result
        return result
