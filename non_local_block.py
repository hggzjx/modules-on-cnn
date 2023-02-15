import tensorflow as tf
from tensorflow.keras import layers

def non_local_block(x, reduction=2, activation=None):
    batch_size, height, width, channels = x.shape.as_list()

    # Compute the feature maps for f, g and h
    f = layers.Conv2D(channels // reduction, 1, activation=activation)(x)
    g = layers.Conv2D(channels // reduction, 1, activation=activation)(x)
    h = layers.Conv2D(channels, 1, activation=activation)(x)

    # Reshape f and g
    f_reshaped = layers.Reshape((height * width, channels // reduction))(f)
    g_reshaped = layers.Reshape((height * width, channels // reduction))(g)

    # Compute the dot product between f and g
    dot_product = layers.Dot(axes=2)([f_reshaped, g_reshaped])

    # Compute the softmax activation
    softmax = layers.Activation('softmax')(dot_product)

    # Compute the dot product between softmax and h
    h_reshaped = layers.Reshape((height * width, channels))(h)
    dot_product2 = layers.Dot(axes=1)([softmax, h_reshaped])

    # Reshape the output and return
    output = layers.Reshape((height, width, channels))(dot_product2)
    return layers.add([x, output])

# # Example usage
# input_shape = (224, 224, 3)
# input_tensor = layers.Input(shape=input_shape)
# output_tensor = non_local_block(input_tensor)

# # Create the model and compile
# model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
# model.compile(optimizer='adam', loss='mse')
