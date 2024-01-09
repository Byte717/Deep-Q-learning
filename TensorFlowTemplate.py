import tensorflow as tf

from tensorflow.python.keras.layers import Dense, LSTM, Conv2D, Conv2DTranspose, Conv1D, Conv1DTranspose
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose, GRU, MaxPool1D, MaxPool2D, MaxPooling3D, AveragePooling2D, AveragePooling1D, AveragePooling3D
from tensorflow.python.keras.layers import GlobalAvgPool1D, GlobalAvgPool2D, GlobalAvgPool3D,GlobalMaxPool1D, GlobalMaxPool2D, GlobalMaxPool3D

from tensorflow.python.keras.activations import relu, softmax, sigmoid, selu, silu, linear, leaky_relu, hard_sigmoid, softsign, softplus, exponential, gelu, swish, tanh, elu

from tensorflow.python.keras.models import Sequential, Functional, save_model, load_model

from tensorflow.python.keras.losses import mean_absolute_error, mean_squared_error, binary_crossentropy, mean_absolute_percentage_error, mean_squared_logarithmic_error,  kl_divergence, log_cosh, huber

from tensorflow.python.keras.optimizers import adam_v2

from tensorboard import *


class MODEL(tf.keras.Model):

    def __init__(self) -> None:
        super().__init__()
