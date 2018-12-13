#!/usr/bin/env python3

from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Model

def set_vgg16_params(m_dict):
    m_dict['num_rows'] = 224
    m_dict['num_cols'] = 224
    m_dict['optimizer_name'] = 'sgd'
    m_dict['learning_rate'] = 1e-3
    m_dict['locked_layers'] = 14
    m_dict['batch_size'] = 8

def get_vgg16_model(m_dict):
    # create the base pre-trained model
    image_shape = (m_dict['num_rows'], m_dict['num_cols'], m_dict['channels'])
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=image_shape)
    
    # create our own top layer
    x = base_model.output
    if False:
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        m_dict['num_top_layers'] = 6
    else:
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        m_dict['num_top_layers'] = 4
    x = Dropout(0.5)(x)
    predictions = Dense(m_dict['num_classes'], activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

