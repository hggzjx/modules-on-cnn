from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, AvgPool2D, Dense, GlobalAvgPool2D, Reshape
from keras.layers import DepthwiseConv2D


def ghost_block(inputs, dw_channels, out_channels, ratio=2):
    init_channels = int(out_channels / ratio)

    # depthwise convolution
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # pointwise convolution
    y = Conv2D(init_channels, kernel_size=1, strides=1, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    # ghost branch
    z = Conv2D(init_channels, kernel_size=1, strides=1, padding='same')(x)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    z = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    z = Conv2D(init_channels, kernel_size=1, strides=1, padding='same')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)

    # concatenate outputs
    output = Concatenate()([y, z])
    output = Conv2D(dw_channels, kernel_size=1, strides=1, padding='same')(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    return output

