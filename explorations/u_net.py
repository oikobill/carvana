"""
Here we make a U-Net for our smaller dataset following the architecture provided https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208

Architecture:

Block 1:                                            # 128
3 --> 16
16 -> 32

maxpool 2 by 2 stride = 2

Block 2:                                            # 64
32 --> 64
64 -> 128

maxpool 2 by 2 stride = 2

Block 3:                                            # 32
128-->256
256 -> 512

maxpool 2 by 2 stride = 2

Block 4:                                            # 16
512-->512
512 -> 512

maxpool 2 by 2 stride = 2

Same:                                               # 8
512-->512

upsample 2 by 2

Upsample 1:                                         # 16
concat(up, block 4, axis = 1)
1024 -> 512
512 --> 512

upsample 2 by 2

Upsample 2:                                         # 32
concat(up, block 3, axis = 3) stack depth-wise
1024 -> 512
512 --> 128

upsample 2 by 2

Upsample 3:                                         # 64
concat(up, block 2, axis = 3) stack depth-wise
256 -> 128
128 --> 32

upsample 2 by 2

Upsample 4:                                         # 128
concat(up, block 1, axis = 3) stack depth-wise
64 -> 64
64 --> 32

Classify:
32 -> 1 + sigmoid

where 
filters in --> filters out is a 3 by 3 convolution with same padding and relu activation
filters in -> filters is a 1 by 1 convolution with same padding
"""
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input
from keras.layers.merge import Concatenate
from keras.models import Model

def unet():
    # input layer 128
    input_layer = Input(shape=(128, 128, 3))

    # Block 1 128
    c1_b1 = Conv2D(filters=16, kernel_size=3, padding="SAME", activation="relu")(input_layer)
    c2_b1 = Conv2D(filters=32, kernel_size=1, padding="SAME", activation="relu")(c1_b1)

    max_pool1 = MaxPooling2D(strides=2)(c2_b1)

    # Block 2 64
    c1_b2 = Conv2D(filters=64, kernel_size=3, padding="SAME", activation="relu")(max_pool1)
    c2_b2 = Conv2D(filters=128, kernel_size=1, padding="SAME", activation="relu")(c1_b2)

    max_pool2 = MaxPooling2D(strides=2)(c2_b2)

    # Block 3 32
    c1_b3 = Conv2D(filters=256, kernel_size=3, padding="SAME", activation="relu")(max_pool2)
    c2_b3 = Conv2D(filters=512, kernel_size=1, padding="SAME", activation="relu")(c1_b3)

    max_pool3 = MaxPooling2D(strides=2)(c2_b3)

    # Block 4 16
    c1_b4 = Conv2D(filters=512, kernel_size=3, padding="SAME", activation="relu")(max_pool3)
    c2_b4 = Conv2D(filters=512, kernel_size=1, padding="SAME", activation="relu")(c1_b4)

    max_pool4 = MaxPooling2D(strides=2)(c2_b4)

    # Same 8
    same = Conv2D(filters=512, kernel_size=3, padding="SAME", activation="relu")(max_pool4)

    # Up 1
    upsample1 = UpSampling2D(2)(same)
    concat1 = Concatenate(-1)([upsample1, c2_b4])

    c1_b5 = Conv2D(filters=512, kernel_size=1, padding="SAME", activation="relu")(concat1)
    c2_b5 = Conv2D(filters=512, kernel_size=3, padding="SAME", activation="relu")(c1_b5)

    # Up 2
    upsample2 = UpSampling2D(2)(c2_b5)
    concat2 = Concatenate(-1)([upsample2, c2_b3])

    c1_b6 = Conv2D(filters=512, kernel_size=1, padding="SAME", activation="relu")(concat2)
    c2_b6 = Conv2D(filters=128, kernel_size=3, padding="SAME", activation="relu")(c1_b6)

    # Up 3
    upsample3 = UpSampling2D(2)(c2_b6)
    concat3 = Concatenate(-1)([upsample3, c2_b2])

    c1_b7 = Conv2D(filters=128, kernel_size=1, padding="SAME", activation="relu")(concat3)
    c2_b7 = Conv2D(filters=32, kernel_size=3, padding="SAME", activation="relu")(c1_b7)

    # Up 4
    upsample4 = UpSampling2D(2)(c2_b7)
    concat4 = Concatenate(-1)([upsample4, c2_b1])

    c1_b8 = Conv2D(filters=128, kernel_size=1, padding="SAME", activation="relu")(concat4)
    c2_b8 = Conv2D(filters=32, kernel_size=3, padding="SAME", activation="relu")(c1_b8)

    out = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(c2_b8)

    model = Model(input_layer, out)

    return model