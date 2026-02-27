import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Reshape, Flatten, Dense, Conv2D, UpSampling2D,
    Activation, BatchNormalization, MaxPooling2D,
    concatenate, Lambda, LeakyReLU, Input,
)
from tensorflow.keras import regularizers, backend as K

def conv_block(prev_layer, filters, use_bn, prefix, conv_size=(5, 5)):
    conv = Conv2D(int(filters), conv_size, padding='same', kernel_initializer="he_normal",  name=prefix + "_conv") (prev_layer)
    #if use_bn: conv = BatchNormalization(name=prefix + "_bn")(conv, training=True)
    conv = LeakyReLU()(conv)
    #conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def create_auto_encoder(filters):
    inputs = Input((256, 192, 5))
    m = filters
    use_bn = False
        
    m *= 2
    c2 = conv_block(inputs, m, use_bn, "conv2_1")
    c2 = conv_block(c2, m, use_bn, "conv2_2")
    p2 = MaxPooling2D((2, 2), name="pool2") (c2)
 
    m *= 2
    c3 = conv_block(p2, m, use_bn, "conv3_1")
    c3 = conv_block(c3, m, use_bn, "conv3_2")
    p3 = MaxPooling2D((2, 2), name="pool3") (c3)
  
    m *= 2
    c4 = conv_block(p3, m, use_bn, "conv4_1")
    c4 = conv_block(c4, m, use_bn, "conv4_2")
    p4 = MaxPooling2D((2, 2), name="pool4") (c4)
 
    m *= 2
    c5 = conv_block(p4, m, use_bn, "conv5_1")
    c5 = conv_block(c5, m, use_bn, "conv5_2")
    p5 = MaxPooling2D((2, 2), name="pool5") (c5)
    
    c6 = conv_block(p5, m, use_bn, "conv6_1")
    c6 = conv_block(c6, m, use_bn, "conv6_2")

    u7 = UpSampling2D() (c6)
    u7 = concatenate([u7, c5])
    c7 = conv_block(u7, m, use_bn, "conv7_1")
    c7 = conv_block(c7, m, use_bn, "conv7_2")

    m /= 2
    u8 = UpSampling2D() (c7)
    u8 = concatenate([u8, c4])
    c8 = conv_block(u8, m, use_bn, "conv8_1")
    c8 = conv_block(c8, m, use_bn, "conv8_2")

    m /= 2
    u9 = UpSampling2D() (c8)
    u9 = concatenate([u9, c3])
    c9 = conv_block(u9, m, use_bn, "conv9_1")
    c9 = conv_block(c9, m, use_bn, "conv9_2")

    m /= 2
    u10 = UpSampling2D() (c9)
    u10 = concatenate([u10, c2])
    c10 = conv_block(u10, m, use_bn, "conv10_1")
    c10 = conv_block(c10, m, use_bn, "conv10_2")

    c11 =  Conv2D(5, (1, 1), activation='linear', name="prediction") (c10) 

    return Model(inputs=[inputs], outputs=[c11])
