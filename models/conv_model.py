import tensorflow as tf

import config as cfg

CONV_WIDTH = 3
OUT_STEPS = 24
# conv_model = tf.keras.Sequential([
#     tf.keras.layers.Conv1D(filters=32,
#                            kernel_size=(CONV_WIDTH,),
#                            activation='relu'),
#     tf.keras.layers.Dense(units=32, activation='relu'),
#     tf.keras.layers.Dense(units=1),
# ])


def create_conv_model(num_features, name='noname'):
    model = tf.keras.Sequential([
        # tf.keras.layers.Lambda(lambda x: x[:, 23:, :], input_shape=(24, num_features)),
        tf.keras.layers.Conv1D(64, activation='relu', kernel_size=3, padding='same',
                               # input_shape=(cfg.prediction['input_len'], num_features)
                               ),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(64, activation='relu', kernel_size=2, padding='same',
                               # input_shape=(cfg.prediction['input_len'], num_features)
                               ),
        tf.keras.layers.MaxPool1D(pool_size=2),

        # tf.keras.layers.Conv1D(512, activation='relu', kernel_size=3, padding='same'),
        # tf.keras.layers.MaxPool1D(pool_size=2),
        # tf.keras.layers.Conv1D(256, activation='relu', kernel_size=3, padding='same'),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(cfg.prediction['num_predictions'], kernel_initializer=tf.initializers.zeros)
    ])
    model.__setattr__('my_name', name)
    model.__setattr__('model_type', 'convolutional')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    return model


def create_conv_model_old(num_features, name='noname'):
    model = tf.keras.Sequential([
        # tf.keras.layers.Lambda(lambda x: x[:, 23:, :], input_shape=(24, num_features)),
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=3, padding='same',
                               # input_shape=(cfg.prediction['input_len'], num_features)
                               ),
        # tf.keras.layers.MaxPool1D(pool_size=2),
        # tf.keras.layers.Conv1D(512, activation='relu', kernel_size=3, padding='same'),
        # tf.keras.layers.MaxPool1D(pool_size=2),
        # tf.keras.layers.Conv1D(256, activation='relu', kernel_size=3, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(cfg.prediction['num_predictions'], kernel_initializer=tf.initializers.zeros)
    ])
    model.__setattr__('my_name', name)
    model.__setattr__('model_type', 'convolutional')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    return model


def create_conv_model_single_input(num_features, name='noname'):
    model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :], input_shape=(24, num_features)),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=CONV_WIDTH),
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(cfg.prediction['num_predictions'], kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        # tf.keras.layers.Reshape([OUT_STEPS, num_features])
        # tf.keras.layers.Reshape([cfg.prediction['num_predictions'], 1])
    ])
    model.__setattr__('my_name', name)
    model.__setattr__('model_type', 'convolutional')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    return model
