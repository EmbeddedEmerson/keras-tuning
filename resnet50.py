#!/usr/bin/env python3

from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Model


def set_resnet50_params(m_dict):
    m_dict['num_rows'] = 224
    m_dict['num_cols'] = 224
    m_dict['optimizer_name'] = 'adam'
    m_dict['learning_rate'] = 1e-4
    m_dict['locked_layers'] = 79

def get_resnet50_model(m_dict):
    # create the base pre-trained model
    image_shape = (m_dict['num_rows'], m_dict['num_cols'], m_dict['channels'])
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=image_shape)
    
    # create our own top layer
    x = base_model.output
    if True:
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
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

	

