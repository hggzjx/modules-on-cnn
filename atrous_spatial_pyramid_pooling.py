from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def atrous_spatial_pyramid_pooling(inputs, rates):
    """
    Atrous Spatial Pyramid Pooling (ASPP) with user-defined rates
    """
    # define atrous convolutional layers with user-defined rates
    conv_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu')(inputs)
    conv_3x3_rates = []
    for rate in rates:
        conv_3x3_rates.append(Conv2D(256, (3, 3), padding='same', dilation_rate=rate, activation='relu')(inputs))

    # global average pooling
    gap = Lambda(lambda x: K.mean(x, axis=[1, 2], keepdims=True))(inputs)
    gap = Conv2D(256, (1, 1), padding='same', activation='relu')(gap)
    gap = Lambda(lambda x: K.squeeze(x, axis=[1, 2]))(gap)

    # concatenate all layers
    concatenated = concatenate([conv_1x1] + conv_3x3_rates + [gap], axis=3)

    return concatenated


# # create the model
# inputs = Input(shape=(None, None, 3))
# rates = [6, 12, 18]  # user-defined rates
# aspp = atrous_spatial_pyramid_pooling(inputs, rates)
# output = Conv2D(1, (1, 1), activation='sigmoid')(aspp)
# model = Model(inputs, output)
