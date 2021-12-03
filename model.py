
# imports for build model
import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Add, Subtract, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

from tensorflow_addons.layers import NoisyDense

# imports for noisy layer
# from tensorflow.keras.layers import Layer
# from tensorflow.keras import activations, initializers, regularizers, constraints
# from tensorflow.keras import backend as K
# import numpy as np


def build_model(state_size, action_size, n_stack, learning_rate):
    input = Input(shape=(state_size[0] // 2, state_size[1] // 2, n_stack))
    x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', padding="same", use_bias=False)(input)
    x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)

    # Split into value and advantage streams
    val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)

    val_stream = Flatten()(val_stream)
    val_stream = NoisyDense(256, kernel_initializer=VarianceScaling(scale=2.), activation="relu", use_bias=True)(val_stream)
    val = NoisyDense(1, kernel_initializer=VarianceScaling(scale=2.), activation="linear", use_bias=True)(val_stream)

    adv_stream = Flatten()(adv_stream)
    adv_stream = NoisyDense(256, kernel_initializer=VarianceScaling(scale=2.), activation="relu", use_bias=True)(adv_stream)
    adv = NoisyDense(action_size, kernel_initializer=VarianceScaling(scale=2.), activation="linear", use_bias=True)(adv_stream)

    # Combine streams into Q-Values
    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))

    output = Add()([val, Subtract()([adv, reduce_mean(adv)])])

    # Build model
    model = Model(input, output)
    model.compile(loss=Huber(), optimizer=Adam(lr=learning_rate))

    return model


# Taken from https://github.com/LuEE-C/NoisyDenseKeras/blob/master/NoisyDense.py
# Paper https://arxiv.org/pdf/1706.10295.pdf

# class NoisyDense(Layer):
#
#     def __init__(self, units,
#                  sigma_init=0.02,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         if 'input_shape' not in kwargs and 'input_dim' in kwargs:
#             kwargs['input_shape'] = (kwargs.pop('input_dim'),)
#         super(NoisyDense, self).__init__(**kwargs)
#         self.units = units
#         self.sigma_init = sigma_init
#         self.activation = activations.get(activation)
#         self.use_bias = use_bias
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.activity_regularizer = regularizers.get(activity_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)
#
#     def build(self, input_shape):
#         assert len(input_shape) >= 2
#         self.input_dim = input_shape[-1]
#
#         self.kernel = self.add_weight(shape=(self.input_dim, self.units),
#                                       initializer=self.kernel_initializer,
#                                       name='kernel',
#                                       regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint)
#
#         self.sigma_kernel = self.add_weight(shape=(self.input_dim, self.units),
#                                       initializer=initializers.Constant(value=self.sigma_init),
#                                       name='sigma_kernel'
#                                       )
#
#
#         if self.use_bias:
#             self.bias = self.add_weight(shape=(self.units,),
#                                         initializer=self.bias_initializer,
#                                         name='bias',
#                                         regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint)
#             self.sigma_bias = self.add_weight(shape=(self.units,),
#                                         initializer=initializers.Constant(value=self.sigma_init),
#                                         name='sigma_bias')
#         else:
#             self.bias = None
#             self.epsilon_bias = None
#
#         self.epsilon_kernel = K.zeros(shape=(self.input_dim, self.units))
#         self.epsilon_bias = K.zeros(shape=(self.units,))
#
#         self.sample_noise()
#         super(NoisyDense, self).build(input_shape)
#
#
#     def call(self, X):
#         perturbation = self.sigma_kernel * self.epsilon_kernel
#         perturbed_kernel = self.kernel + perturbation
#         output = K.dot(X, perturbed_kernel)
#         if self.use_bias:
#             bias_perturbation = self.sigma_bias * self.epsilon_bias
#             perturbed_bias = self.bias + bias_perturbation
#             output = K.bias_add(output, perturbed_bias)
#         if self.activation is not None:
#             output = self.activation(output)
#         return output
#
#     def compute_output_shape(self, input_shape):
#         assert input_shape and len(input_shape) >= 2
#         assert input_shape[-1]
#         output_shape = list(input_shape)
#         output_shape[-1] = self.units
#         return tuple(output_shape)
#
#     def sample_noise(self):
#         K.set_value(self.epsilon_kernel, np.random.normal(0, 1, (self.input_dim, self.units)))
#         K.set_value(self.epsilon_bias, np.random.normal(0, 1, (self.units,)))
#
#     def remove_noise(self):
#         K.set_value(self.epsilon_kernel, np.zeros(shape=(self.input_dim, self.units)))
#         K.set_value(self.epsilon_bias, np.zeros(shape=self.units,))
