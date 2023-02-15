from keras.layers import Input, Conv2D, Concatenate, Activation
from keras.models import Model

def adaptively_spatial_feature_fusion(inputs):
    convs = []
    for i, x in enumerate(inputs):
        convs.append(Conv2D(filters=1, kernel_size=1, padding='same', name='conv_%s' % i)(x))
    weights = Concatenate(name='concat')(convs)
    weights = Activation('sigmoid', name='sigmoid')(weights)
    out = []
    for i, x in enumerate(inputs):
        out.append(x * weights[:, :, :, i:i+1])
    out = Concatenate(name='out_concat')(out)
    out = Activation('relu', name='out_relu')(out)
    return out
