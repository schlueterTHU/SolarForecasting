import tensorflow as tf

import config as cfg


def create_lstm_conv_model(num_features, name='noname'):
    model= tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        # tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Conv1D(filters=256, activation='relu', kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(cfg.prediction['num_predictions'], kernel_initializer=tf.initializers.zeros)
        # tf.keras.layers.Dense(cfg.prediction['num_predictions'] * num_features/12,
                              # kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        # tf.keras.layers.Reshape([cfg.prediction['num_predictions'], num_features])
    ])

    model.__setattr__('my_name', name)
    model.__setattr__('model_type', 'lstm_conv')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    return model
