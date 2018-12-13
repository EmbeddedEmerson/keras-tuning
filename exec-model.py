#!/usr/bin/env python3

#
#	exec-model.py
#
#   Driver code for fine tuning Keras applications.
#

import keras
from keras import backend as K
from keras import metrics
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy
from my_classes import DataGenerator
from my_utils import print_elapsed_time, generate_dataset, prepare_save_directory, \
     set_model_params, log_model_details, get_optimizer
from vgg16 import get_vgg16_model
from resnet50 import get_resnet50_model
from inceptionv3 import get_inceptionv3_model
import numpy as np
import time
import os


model_filepath=None
model_params = {}       # dictionary of model parameters
train_list = []
test_list = []

# for convenience, the code relies on the values stored in model_params dict
model_name = 'resnet50'          # one of'vgg16', 'resnet50', or 'inceptionv3'
rev = 'rev-6-f'
run_mode = 'noop'              # 'train-eval', 'train', 'eval', 'summary', 'noop'

def get_compiled_model():
    if os.path.isfile(model_filepath):
        print('\nLoading model weights from disk')
        K.set_learning_phase(1)
        print('  setting learning phase to 1 prior to creating model and loading weights')
        model = create_model()
        model.load_weights(model_filepath)
    else:
        print('\nCreating model from scratch')
        model = create_model()
    print('  model ready for use')
    return model

def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

def top_1_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)

def create_model():
    name = model_params['model_name']
    if name == 'vgg16':
        model = get_vgg16_model(model_params)
    elif name == 'resnet50':
        model = get_resnet50_model(model_params)
    elif name == 'inceptionv3':
        model = get_inceptionv3_model(model_params)
    else:
        assert False, 'fatal error, create_model(), model_name: ' + name + ' invalid'

    # set the indicated number of layers to non-trainable
    locked = model_params['locked_layers']
    for layer in model.layers[:locked]:
        layer.trainable = False
	# make certain the remaining layers are trainable
    for layer in model.layers[locked:]:
        layer.trainable = True

    # instantiate desired optimizer and compile model
    optimizer = get_optimizer(model_params)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', \
                  metrics=[top_1_accuracy, top_5_accuracy])
    return model

#------------------------------------------------------------------------------------------
#
#	Program entry point.
#
#

# init start time
start_time = time.monotonic()

# transfer model, revision and run mode to the dictionary
model_params['model_name'] = model_name
model_params['rev'] = rev
model_params['run_mode'] = run_mode    

# specify remaining model parameters
set_model_params(model_params)

# prepare directory and file path for saving model
model_filepath = prepare_save_directory(model_params)

# get the model
model = get_compiled_model()

# load datasets
(partition_dict, labels_dict) = generate_dataset(model_params['num_classes'])

# build the training and test lists
for partition in model_params['train_partitions']:
    train_list += partition_dict[partition]
for partition in model_params['test_partitions']:
    test_list += partition_dict[partition]
model_params['train_len'] = len(train_list)
model_params['test_len'] = len(test_list)

# specify generator parameters
params = {'dim': (model_params['num_rows'],model_params['num_cols']),
          'batch_size': model_params['batch_size'],
          'n_classes': model_params['num_classes'],
          'n_channels': model_params['channels'],
          'shuffle': True,
          'scale_image': model_params['preprocess_input'] }

# instantiate the generators
train_generator = DataGenerator(train_list, labels_dict, model_params['augmentation'], **params)
test_generator = DataGenerator(test_list, labels_dict, False, **params)

# log details
log_model_details(model, model_params)

# for convenience
model_rev_name = model_params['model_name'] + ' ' + model_params['rev']
mode = model_params['run_mode']

if mode == 'train' or mode == 'train-eval':
    # train the model on the new data for specified epochs
    model.fit_generator(generator=train_generator,
                        epochs=model_params['training_epochs'],
                        use_multiprocessing=True,
                        workers=8)
    
    if mode == 'train-eval':
        # evaluate trained model
        print('\nEvaluate ' + model_rev_name + ' model' )
        scores = model.evaluate_generator(generator=test_generator, 
                                          use_multiprocessing=True,
                                          workers=8,
                                          verbose=1)
        print('  Test loss:', scores[0])
        print('  Top-1 accuracy:', scores[1])
        print('  Top-5 accuracy:', scores[2])

    # save model and weights to disk
    print('Saving ' + model_rev_name + ' weights to disk')
    model.save_weights(model_filepath)
    print('  save operation complete')
elif mode == 'eval':
    # evaluate trained model
    print('\nEvaluate ' + model_rev_name + ' model' )
    scores = model.evaluate_generator(generator=test_generator, 
                                      use_multiprocessing=True,
                                      workers=8,
                                      verbose=1)
    print('  Test loss:', scores[0])
    print('  Top-1 accuracy:', scores[1])
    print('  Top-5 accuracy:', scores[2])
elif mode == 'summary':
    print(model.summary())
elif mode == 'noop':
    mode = 'noop'
else:
    assert False, 'unrecognized run_mode'


# display elapsed time
elapsed_time = time.monotonic() - start_time
print_elapsed_time(elapsed_time, 'Training elapsed time: ')


