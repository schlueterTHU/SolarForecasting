import tensorflow as tf
import config as cfg


class RepeatBaseline(tf.keras.Model):
    def __init__(self, label_index=None, name='noname'):
        super().__init__()
        self.label_index = label_index
        self.__setattr__('my_name', name)
        self.__setattr__('model_type', 'naive')

    def call(self, inputs):
        return inputs[:, :, cfg.prediction['pos']]

    def compile_baseline(self):
        self.compile(loss=tf.losses.MeanSquaredError(),
                     metrics=[tf.metrics.MeanAbsoluteError()])