import tensorflow as tf

from models.baseline import RepeatBaseline
# import models.linear_model as lm
import models.conv_model as conv
import models.lstm_model as lstm
# import models.dense_model as dm
# from models.autoregressive_model import FeedBack
import models.convLSTM_model as convLSTM
import models.LSTMconv_model as LSTMconv

import config as cfg


def choose_model(model):
    num_features = model.get('num_features', 1)
    name = model.get('name', '')
    model_dict = {
        'lstm': lstm.create_lstm_model,
        'convolutional': conv.create_conv_model,
        'conv_lstm': convLSTM.create_conv_lstm_model,
        'lstm_conv': LSTMconv.create_lstm_conv_model,
        'naive': lambda x, y: RepeatBaseline(),
        # 'test': lstm.create_lstm_model_test
    }
    func = model_dict.get(model['type'], lambda x, y: "Invalid model type")
    return func(num_features, name)


def fit_model(model, window):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=cfg.training['patience'],
                                                      mode='min')

    history = model.fit(window.train, epochs=cfg.training['max_epochs'],
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


def build_model(model, window, path, train=False):
    if model.__getattribute__('model_type') != 'naive':
        if train:
            fit_model(model, window)
            model.save_weights(path)
        else:
            model.load_weights(path).expect_partial()
            # model.build((None, 34, 7))
    else:
        model.compile_baseline()
    return model


def get_predictions(model, data, label, scaler):
    predictions = model(data).numpy()
    return scaler.inverse_transform(correct_values(predictions, label))
    # return correct_values(scaler.inverse_transform(predictions), label)
    # return scaler.inverse_transform(correct_values(predictions[:, :, 0], label))


def correct_values(pred, label):
    pred[label == 0] = 0
    pred[pred < 0] = 0
    return pred
