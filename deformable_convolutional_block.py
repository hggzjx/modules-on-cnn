import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class DeformableConv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', dilation_rate=(1, 1), groups=1, deformable_groups=1, **kwargs):
        super(DeformableConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.deformable_groups = deformable_groups

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(*self.kernel_size, input_shape[-1] // self.groups, self.filters), initializer='glorot_uniform', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros', trainable=True)
        offset_shape = (*self.kernel_size, input_shape[-1] // self.groups, 2 * self.kernel_size[0] * self.kernel_size[1] * self.deformable_groups)
        self.offset_weight = self.add_weight(name='offset_weight', shape=offset_shape, initializer='zeros', trainable=True)
        self.offset_bias = self.add_weight(name='offset_bias', shape=(offset_shape[-1],), initializer='zeros', trainable=True)
        super(DeformableConv2D, self).build(input_shape)

    def call(self, inputs):
        offset = K.conv2d(inputs, self.offset_weight, strides=self.strides, padding=self.padding, dilation_rate=self.dilation_rate, groups=self.deformable_groups)
        offset = K.bias_add(offset, self.offset_bias)
        output = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, dilation_rate=self.dilation_rate, groups=self.groups)(inputs)
        output = K.permute_dimensions(output, [0, 3, 1, 2])
        offset = K.permute_dimensions(offset, [0, 3, 1, 2])
        output = K.reshape(output, (output.shape[0] * output.shape[1], 1, output.shape[2], output.shape[3]))
        offset = K.reshape(offset, (offset.shape[0] * offset.shape[1], 2 * self.kernel_size[0] * self.kernel_size[1], offset.shape[2], offset.shape[3]))
        output = K.reshape(K.tf.nn.deformable_conv2d(output, offset, self.kernel, strides=self.strides, padding=self.padding, dilations=self.dilation_rate, groups=self.groups, deformable_groups=self.deformable_groups), (inputs.shape[0], self.filters, inputs.shape[2], inputs.shape[3]))
        output = K.bias_add(output, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.filters
        return tuple(output_shape)