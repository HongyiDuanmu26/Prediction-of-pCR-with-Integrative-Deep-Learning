from __future__ import print_function
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, multiply, Dense, Dropout, Flatten, GaussianNoise, AlphaDropout, ReLU, Conv2D, BatchNormalization
from keras.activations import selu


def CustomNet():
	image_input=Input((256,256,60),name='image_input')
        clinic_input=Input((18,),name='clinical_input')

################ clinical side

        co1=Dense(64,activation='selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal')(clinic_input)
        co1=AlphaDropout(0.1)(co1)

        co2=Dense(64,activation='selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal')(co1)
        co2=AlphaDropout(0.1)(co2)

        co3=Dense(128,activation='selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal')(co2)
        co3=AlphaDropout(0.1)(co3)

        co4=Dense(256,activation='selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal')(co3)
        co4=AlphaDropout(0.1)(co4)

        co5=Dense(256,activation='selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal')(co4)
        co5=AlphaDropout(0.1)(co5)

############### image side

        output=Conv2D(64, (3, 3), padding='same',use_bias=False,kernel_initializer=keras.initializers.he_normal(seed=122),kernel_regularizer=regularizers.l2(0.001))(image_input)
        output=BatchNormalization(axis=-1,center=False)(output)
        output=ReLU()(output)

        output=Conv2D(64, (3, 3), strides=(2,2), padding='same',use_bias=False,kernel_initializer=keras.initializers.he_normal(seed=123),kernel_regularizer=regularizers.l2(0.001))(output)
        output=multiply([output,co1])
        output=BatchNormalization(axis=-1,center=False)(output)
        output=ReLU()(output)

##############
        output=Conv2D(64, (3, 3), padding='same',use_bias=False,kernel_initializer=keras.initializers.he_normal(seed=124),kernel_regularizer=regularizers.l2(0.001))(output)
        output=BatchNormalization(axis=-1,center=False)(output)
        output=ReLU()(output)

        output=Conv2D(64, (3, 3), strides=(2,2),padding='same',use_bias=False,kernel_initializer=keras.initializers.he_normal(seed=125),kernel_regularizer=regularizers.l2(0.001))(output)
        output=multiply([output,co2])
        output=BatchNormalization(axis=-1,center=False)(output)
        output=ReLU()(output)

##############
        output=Conv2D(128, (3,3), padding='same',use_bias=False,kernel_initializer=keras.initializers.he_normal(seed=126))(output)
        output=BatchNormalization(axis=-1,center=False)(output)
        output=ReLU()(output)

        output=Conv2D(128, (3, 3), strides=(2,2), padding='same',use_bias=False,kernel_initializer=keras.initializers.he_normal(seed=127))(output)
        output=multiply([output,co3])
        output=BatchNormalization(axis=-1,center=False)(output)
        output=ReLU()(output)
############
        output=Conv2D(256, (3, 3), padding='same',use_bias=False,kernel_initializer=keras.initializers.he_normal(seed=129))(output)
        output=BatchNormalization(axis=-1,center=False)(output)
        output=ReLU()(output)

        output=Conv2D(256, (3, 3), strides=(2,2), padding='same',use_bias=False,kernel_initializer=keras.initializers.he_normal(seed=130))(output)
        output=multiply([output,co4])
        output=BatchNormalization(axis=-1,center=False)(output)
        output=ReLU()(output)
#############
        output=Conv2D(256, (3, 3), padding='same',use_bias=False,kernel_initializer=keras.initializers.he_normal(seed=132))(output)
        output=BatchNormalization(axis=-1,center=False)(output)
        output=ReLU()(output)

        output=Conv2D(256, (3, 3), strides=(2,2), padding='same',use_bias=False,kernel_initializer=keras.initializers.he_normal(seed=133))(output)
        output=multiply([output,co5])
        output=BatchNormalization(axis=-1,center=False)(output)
        output=ReLU()(output)
##############

        output=Flatten()(output)
        output=Dense(4096)(output)
        output=ReLU()(output)


        output=Dense(512)(output)
        output=ReLU()(output)
        output=Dense(64)(output)
        output=ReLU()(output)


        output=Dense(2)(output)
        output=Activation('softmax')(output)

        model=Model(inputs=[image_input,clinic_input],outputs=output)

        return model 

