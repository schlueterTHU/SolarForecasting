import tensorflow as tf

import config as cfg


def create_conv_lstm_model(num_features, name='noname'):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=256, activation='relu', kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.Conv1D(filters=256, activation='relu', kernel_size=2, strides=1, padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(cfg.prediction['num_predictions'],
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        # tf.keras.layers.Reshape([cfg.prediction['num_predictions'], num_features])
    ])

    model.__setattr__('my_name', name)
    model.__setattr__('model_type', 'conv_lstm')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    return model


def create_convLSTM_model_old():
    model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        # tf.keras.layers.Lambda(lambda x: x[:, -3:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(filters=8, activation='relu', kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.Conv1D(filters=8, activation='relu', kernel_size=2, strides=1, padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=2),

        tf.keras.layers.Conv1D(filters=16, activation='relu', kernel_size=2, strides=1, padding='same'),
        tf.keras.layers.Conv1D(filters=16, activation='relu', kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=2),

        tf.keras.layers.Conv1D(filters=32, activation='relu', kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.Conv1D(filters=64, activation='relu', kernel_size=2, strides=1, padding='same'),

        tf.keras.layers.LSTM(24, return_sequences=False, dropout=0.2),
        # tf.keras.layers.LSTM(24, return_sequences=False, dropout=0.2),

        tf.keras.layers.Dense(512, kernel_initializer=tf.initializers.zeros),
        tf.keras.layers.Dense(64, kernel_initializer=tf.initializers.zeros),
        tf.keras.layers.Dense(24, kernel_initializer=tf.initializers.zeros),

        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([24, 1])
    ])
    model.__setattr__('my_name', 'ConvLSTM')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(learning_rate=0.00005),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    return model
