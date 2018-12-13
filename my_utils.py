#!/usr/bin/env python3

#
#   my_utils.py
#
#   Collection of miscellaneous utilities.
#

from keras.optimizers import SGD, Adam
from vgg16 import set_vgg16_params
from resnet50 import set_resnet50_params
from inceptionv3 import set_inceptionv3_params
import time
import numpy as np
import os

txt_prefix = '../data/'
class_dict = {}
partition_dict = {}
labels_dict = {}

def populate_class_dict(dict_fname):
    dict_filepath = txt_prefix + dict_fname
    fh = None
    try:
        fh = open(dict_filepath, encoding='utf8')
        for index, line in enumerate(fh):
            variant = line.strip()
            class_dict[variant] = index
    finally:
        if fh is not None:
            fh.close()


#
#   Given the name of a text file containing (image_name, variant)
#   pairs, perform the following:
#       add image_name to local image_list
#       update global labels dictionary with,
#           key=image_name, value=class index of(variant)
#
#   On exit, return the populated image_list
#

def populate_image_list(image_fname):
    image_filepath = txt_prefix + image_fname
    image_list = []
    fh = None
    try:
        fh = open(image_filepath, encoding='utf8')
        for index, line in enumerate(fh):
            # formulate image_name and variant
            (image_name, remainder) = line.split(' ',1)
            variant = remainder.strip()

            # make sure variant is a member of the class dict
            class_index = class_dict.get(variant)
            assert class_index != None, 'variant not found in class dictionary'

            # add image_name to list
            image_list.append(image_name)

            # add (image_name, class_index) to the labels dict
            labels_dict[image_name] = class_index
    finally:
        if fh is not None:
            fh.close()
    return image_list


def generate_dataset(num_classes):
    print('\nGenerating dataset')
    # populate the class dict and error check
    print('  Populating class dict')
    populate_class_dict('variants.txt')
    assert len(class_dict) == num_classes, 'Unexpected class dictionary length'

    partition_list = [('train', 'images_variant_train.txt'),
                      ('val', 'images_variant_val.txt'),
                      ('test', 'images_variant_test.txt')]

    print('  Populating partition and labels dicts')
    for index, value in enumerate(partition_list):
        (partition_name, list_fname) = value
        temp_list = populate_image_list(list_fname)
        print('    partition name: ' + partition_name + ', list length: ' + str(len(temp_list)))
        partition_dict[partition_name] = temp_list

    print('  Results:')
    print('    partition_dict len: ' + str(len(partition_dict)))
    print('    labels_dict len: ' + str(len(labels_dict)))
    print('    class_dict len: ' + str(len(class_dict)))

    return partition_dict, labels_dict


def print_elapsed_time(elapsed, label=None):
    # calculate seconds and milliseconds
    seconds = int(elapsed)
    float_remainder = elapsed - seconds
    float_milliseconds = float_remainder * 1000
    milliseconds = int(float_milliseconds)

    # generate formatted string
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    h = str(h).rjust(2,'0') #covert to strings,
    m = str(m).rjust(2,'0') #adding 0 if necessary to make 
    s = str(s).rjust(2,'0') #each one two digits long
    ms = str(milliseconds).rjust(3,'0')
    time_string =  "{}:{}:{}.{}".format(h,m,s,ms)

    # display result
    if label == None:
        print('Elapsed time: ' + time_string + '\n')
    else:
        print(label + time_string + '\n')

def prepare_save_directory(m_dict):
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    full_name = m_dict['model_name'] + '-' + m_dict['rev'] + '.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, full_name)
    return file_path

def get_optimizer(m_dict):
    name = m_dict['optimizer_name']
    rate = m_dict['learning_rate']
    if name == 'adam':
        optimizer = Adam(lr=rate)
    elif name == 'sgd':
        optimizer = SGD(lr=rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        assert False, 'fatal error, optimizer: ' + name + ' not recognized'
    return optimizer

def default_preprocess_input(X):
    X = X/255
    return X

def set_model_params(m_dict):
    m_dict['training_epochs'] = 64
    m_dict['channels'] = 3
    m_dict['num_classes'] = 100
    m_dict['batch_size'] = 16
    m_dict['augmentation'] = True
    m_dict['train_partitions'] = ['train', 'val']
    m_dict['test_partitions'] = ['test']
    m_dict['preprocess_input'] = default_preprocess_input

    name = m_dict['model_name']
    if name == 'vgg16':
        set_vgg16_params(m_dict)
    elif name == 'resnet50':
        set_resnet50_params(m_dict)
    elif name == 'inceptionv3':
        set_inceptionv3_params(m_dict)
    else:
        assert False, 'fatal error, set_model_params(), model_name: ' + name + ' invalid'

def log_top_architecture(model, m_dict):
    start_index = len(model.layers) - m_dict['num_top_layers']
    print('  Top Architecture:')
    for layer in model.layers[start_index:]:
        print('    ' + layer.name + '\t' + str(layer.output_shape))

def log_model_details(model, m_dict):
    image_shape = (m_dict['num_rows'], m_dict['num_cols'], m_dict['channels'])
    print('\nModel Details')
    print('  Name: ' + m_dict['model_name'] + ' ' + m_dict['rev'])
    print('  Total layers: ' + str(len(model.layers)))
    print('  Non-trainable layers: ' + str(m_dict['locked_layers']))
    log_top_architecture(model, m_dict)
    print('  Number classes: ' + str(m_dict['num_classes']))
    print('  Optimizer : ' + m_dict['optimizer_name'])
    print('    Learning rate: ' + str(m_dict['learning_rate']))
    print('  Image shape: ' + str(image_shape))
    print('  Training partitions: ' + str(m_dict['train_partitions']))
    print('  Training images: ' + str(m_dict['train_len']))
    print('  Test partitions: ' + str(m_dict['test_partitions']))
    print('  Test images: ' + str(m_dict['test_len']))
    print('  Augmentation: ' + str(m_dict['augmentation']))
    print('  Batch size: ' + str(m_dict['batch_size']))
    print('  Epochs: ' + str(m_dict['training_epochs']))
    print('  Run mode: ' + m_dict['run_mode'])
    print()






