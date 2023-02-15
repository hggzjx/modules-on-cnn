from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape, Dense, multiply
from tensorflow.keras.models import Model
'''    
ratio用于控制Squeeze层中全连接层的输出通道数。一般情况下，ratio的取值范围为8到16之间。
该超参数的选择对Squeeze-and-Excitation Networks的性能和计算复杂度都有一定的影响。较小的ratio会减少Squeeze层中全连接层的参数数量，从而降低模型的计算复杂度，但可能会损失一些模型性能；较大的ratio会增加Squeeze层中全连接层的参数数量，从而提高模型的性能，但也会增加模型的计算复杂度。
'''
def squeeze_excite_block(input_tensor, ratio=16):

    # 获取输入张量的通道数
    channels = input_tensor.shape[-1]
    
    # 计算Squeeze层的输出张量形状
    se_shape = (1, 1, channels)
    
    # 添加Squeeze层
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    
    # 添加两个全连接层，并使用ReLU激活函数
    se = Dense(channels // ratio, activation='relu', kernel_initializer='he_normal')(se)
    se = Dense(channels, activation='sigmoid', kernel_initializer='he_normal')(se)
    
    # 使用乘法运算符将Squeeze层的输出与输入张量相乘
    x = multiply([input_tensor, se])
    
    return x

# def create_model(input_shape):
#     inputs = Input(shape=input_shape)
    
#     # 添加卷积层和池化层
#     x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
#     x = squeeze_excite_block(x)
#     x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = squeeze_excite_block(x)
#     x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = squeeze_excite_block(x)
#     x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = squeeze_excite_block(x)
    
#     # 添加全局平均池化层和输出层
#     x = GlobalAveragePooling2D()(x)
#     outputs = Dense(10, activation='softmax')(x)
    
#     # 创建模型
#     model = Model(inputs=inputs, outputs=outputs)
    
#     return model

# # 创建模型
# input_shape = (None, 13, 35, 768)
# model = create_model(input_shape)
