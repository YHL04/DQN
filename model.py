
# imports for build model
import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Add, Subtract, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

# imports for noisy nets
from tensorflow_addons.layers import NoisyDense


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

