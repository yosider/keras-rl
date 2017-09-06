from keras.models import Model
from keras.layers import Inputs, BatchNormalization, Convolution2D, Dense, Flatten
from keras import backend as K

def build_visual_model():
    input_shape = (3, 100, 100)
    inputs = Inputs(shape=input_shape)
    # inputs_ = BatchNormalization(inputs)
    v1 = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(inputs_)
    v1_ = BatchNormalization(v1)
    v2 = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(v1_)
    v2_ = BatchNormalization(v2)
    # v3->v5 & v4->v6: parallel
    v3 = Convolution2D(32, 2, 2, border_mode='same', activation='relu')(v2_)
    v3_ = BatchNormalization(v3)
    v5 = Convolution2D(16, 2, 2, border_mode='same', activation='relu')(v3_)
    v5_ = BatchNormalization(v5)
    output_what = Flatten(v5_)

    v4 = Convolution2D(32, 2, 2, border_mode='same', activation='relu')(v2_)
    v4_ = BatchNormalization(v4)
    v6 = Convolution2D(16, 2, 2, border_mode='same', activation='relu')(v4_)
    v6_ = BatchNormalization(v6)
    output_where = Flatten(v6_)

    model = Model(input=inputs, output=[output_what, output_where])
    return model
