import tensorflow as tf

import config as cfg


def create_lstm_model(num_features, name='noname'):
    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(32, return_sequences=False, input_shape=(24, num_features)),
        # Shape => [batch, out_steps*features]
        # tf.keras.layers.Dense(cfg.prediction['num_predictions'] * num_features,
        #                       kernel_initializer=tf.initializers.zeros),
        tf.keras.layers.Dense(cfg.prediction['num_predictions'], kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        # tf.keras.layers.Reshape([cfg.prediction['num_predictions'], num_features])
        # tf.keras.layers.Reshape([cfg.prediction['num_predictions'], 1])
    ])
    model.__setattr__('my_name', name)
    model.__setattr__('model_type', 'lstm')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(cfg.training['learning_rate']),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    return model
