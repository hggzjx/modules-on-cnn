from keras.layers import Conv2D, Activation, BatchNormalization, Concatenate
from keras import backend as K

def conv_block(inputs, filters, kernel_size, strides=(1, 1), padding='same', dilation_rate=(1, 1)):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def receptive_field_block(inputs, out_channels, stride=1):
    # 3x3 conv
    branch1 = conv_block(inputs, out_channels // 4, (1, 1))
    branch1 = conv_block(branch1, out_channels // 4, (3, 3), strides=(stride, stride))

    # 5x5 conv
    branch2 = conv_block(inputs, out_channels // 4, (1, 1))
    branch2 = conv_block(branch2, out_channels // 4, (3, 3))
    branch2 = conv_block(branch2, out_channels // 4, (3, 3), strides=(stride, stride))

    # max pooling
    branch3 = K.max_pooling2d(inputs, pool_size=(3, 3), strides=stride, padding='same')
    branch3 = conv_block(branch3, out_channels // 4, (1, 1))

    x = Concatenate()([branch1, branch2, branch3])
    x = conv_block(x, out_channels, (1, 1))
    return x

