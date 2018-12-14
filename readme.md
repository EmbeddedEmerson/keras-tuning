# readme.md #

## Introduction ##

Recently, I've been studying machine learning.  As part of the coursework, I gained an appreciation for using published, pre-trained models as the starting point for new work.  In particular, fine tuning existing models looks like a promising approach.  This strategy is also called Transfer Learning.

This project takes an existing dataset (10,000 labeled images of airplanes) and fine tunes existing Keras models to perform image recognition.

## Dataset ##

The FGVC-Aircraft Benchmark (Fine-Grained Visual Classification of Aircraft, S. Maji, J. Kannala, E. Rahtu, M. Blaschko, A. Vedaldi, arXiv.org, 2013) is the dataset for my project.  A complete description can be found here:

    http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

This dataset was used for a 2013 fine-grained recognition challenge.  The fact that this dataset was composed by professionals removes significant sources of error and allowed me to focus exclusively on the recognition problem.  Although the data is organized into three tiers (variant, family, manufacturer), the models below are tuned to recognize variants only.

With a little bit of guidance most people can be trained to differentiate between a Boeing 747 and a Boeing 737.  In the nomenclature of the dataset, Boeing 747 and Boeing 737 are examples of families.  Within an individual family, Boeing 737 for example, there are multiple variants: 737-200, 737-400, 737-900, etc.  Normally, the relative length of the fuselage is a good clue for determining variant.  However, the differences can be subtle - fertile ground for machine learning. The dataset contains 100 different variants; there are 100 different examples of each variant.  Hence 10,000 images total.  

Broadly speaking, the dataset is divided into three partitions:
- train, 3334 images
- validation, 3333 images
- test, 3333 images

For the purposes of this project, I combined the train and validation partitions and used them for training the models.  I used the test partition as is for evaluating the performance of the models.  To summarize:

* training set, 6667 images
* test set, 3333 images

At no point in time were the models ever trained using the test data; the test data was only used for evaluating performance.


## Development Environment ##

I began this project on an Ubuntu PC without a GPU.  After gaining some traction, I realized the training times were too long to realistically make progress.  For example, some of my early non-GPU training runs were taking approximately one hour per epoch.  This was unacceptible and I invested in an Nvidia GeForce 1060 GPU.  Installing the GPU and associated software was a minor project in itself.  Below are a few details.

My PC runs an AMD® Athlon(tm) ii 255 dual-core processor.  This is an older CPU and newer versions of TensorFlow will crash because the processor does not support instructions present in the binary installs.  Specifically, later versions of TensorFlow install binaries which assume the underlying CPU supports AVX instructions; my Athlon CPU does not. 

Using the recommended 'hello world' tensorflow test program, I determined that TensorFlow 1.5 was the most recent version that would run properly on my machine.  A second option would have been downloading the latest TensorFlow source, and then compiling and installing it on my machine.  I chose the version 1.5 option to reduce the number of variables I was working with.

Below are the specifics on my development environment:

    PC running Ubuntu 18.04 LTS, 64-bit.  12 GiB memory, 500 watt power supply.
    Nvidia GeForce GTX 1060 GPU, 6 GiB memory.
    tensorflow-gpu        1.5.0
    CUDA                  9.0.176
    cuDNN                 7.0.5
    Nvidia driver         390.87
    Keras                 2.2.4
    Keras-Applications    1.0.6
    Keras-Preprocessing   1.0.5
    Python                3.6.5

Notes:
1. There are version dependencies between tensorflow-gpu, CUDA, cuDNN, and Nvidia driver components.
2. Following the Google recommendation, tensorflow and Keras components installed and executed using a virtual environment.


## Preparing the Dataset ##

Download the 'Data, annotations and evaluation' archive from the above fgvc-aircraft link.  Extracting the archive yields the following directory structure:

    fgvc-aircraft-2013b/
        evaluation, README and vl files, ignored
        data/
            variants.txt
            images_box.txt
            images_variant_train.txt, images_variant_val.txt, images_variant_test.txt
            family and manufacturer .txt files - ignored
            images/
                10,000 .jpg files, each containing an aircraft image

### Variants ###

Variants.txt is a list of the 100 aircraft variants found in the dataset. Below is a snippet which enumerates the Boeing 737 variants:

    737-200
    737-300
    737-400
    737-500
    737-600
    737-700
    737-800
    737-900

### Bounding Box ###

For each aircraft found in the images directory, images_box.txt specifies the bounding box for that image.  Sample line:

    1340192 83 155 964 462

This means that the bounding box for the image file 1340192.jpg has the coordinates,
    upper left corner (83,155)
    lower right corner (964,462)
As you would guess, images_box.txt is used for image preprocessing.  See image_boxer.py below.

### Partition Members ###

Images_variant_train, images_variant_val and images_variant_test .txt files list the members of the train, validation and test partitions.  Each line from the file is an (image name,variant) pair.  For example:

    1042824 707-320

In other words, the image file 1042824.jpg is a picture of the Boeing 707-320 aircraft variant.


## Preparing the git archive ##

If you desire to experiment with the code and dataset, clone the archive into the

    fgvc-aircraft-2013b/

directory.  Ignoring the details of the dataset, this produces the following directory structure:

    fgvc-aircraft-2013b/
        data/
        keras-tuning/
            image_boxer.py
            exec-model.py
            my_utils.py
            my_classes.py
            vgg16.py
            resnet50.py
            inceptionv3.py
            logs/
                 log files from train and eval runs
                 model summaries
            saved_models/
                 empty, populated as a by-product of training

Sections below elucidate each one of these components.


## Components ##

### Boxed Images ###

The training/evaluation code assumes that there is a fgvc-aircraft-2013b/data/boxed directory containing the boxed image of each aircraft found in the fgvc-aircraft-2013b/data/images directory.  This is a preprocessing step to cut execution time.

To experiment with the code, copy image_boxer.py to fgvc-aircraft-2013b/data, create the fgvc-aircraft-2013b/data/boxed directory, then run image_boxer.py.  Using the contents of images_box.txt, the utility appropriately populates the boxed directory.

### Data Generators ###

I started out on this project using arrays to hold the aircraft images.  It didn't take very long to figure out that all of my dataset couldn't be stored in memory.  Besides, a dataset containing 10K examples isn't considered large.  How were other people handling truly large datasets?

Data generators are a Keras mechanism for loading and feeding data into a model for training or evaluation.  The generator supplies data in batches.  Initially, I looked into ImageDataGenerator and its associated flow_from_directory() method.  The flow method requires a specific directory structure, namely a separate directory for each class.  My dataset has 100 classes (100 variants) so this seemed a bit impractical.

After further research, I hit upon the following link:

    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

The article describes a DataGenerator which removes the directory structure limitation; its structure is purposefully generic.  One tailors this code via: 

* User supplied train, val and test lists
* Specifying the details for loading an individual image
* User supplied labels dictionary

Other than some trivial modifications, this is not my code.  See my_classes.py for details.

### Tailoring the DataGenerator ###

Refer to the above stanford link.  The lists needed for training, etc. are first organized into the partition dictionary:

* partition_dict['train'] - list of IDs composing the training set
* partition_dict['val'] - list of IDs composing the validation set
* partition_dict['test'] - list of IDs composing the test set

In the context of the fgvc dataset, each ID is the name of an image file with the .jpg extension omitted. So, the train partion will contain a list of all the .jpg files used for training.  Likewise for the val and test partitions.

Refer to get_image() in my_classes.py.  You can see how given an image id ('1042824'), it is easy to load the corresponding .jpg file from the boxed/ directory, convert the resulting image to a Python array, and then use its contents for further processing.

A labels dictionary is the third component needed for tailoring the DataGenerator. The keys for the dictionary are image ids ('1042824'). The value for a given key is its class index.  Class index is a number range 0 thru 99 inclusive, which identifies the aircraft variant found in the associated image file.

Code in my_utils.py populates these structures at program start-up.  Code in exec_model.py combines the train and val lists into a single training list containing 6667 members.

### Data Augmentation ###

As stated above, this project uses 6667 images for model training, and the remaining 3333 images for model evaluation.  For a model with 100 classes, this is not that much data.  How can you train it without overfitting?

Data augmentation provides the answer.  This technique alters the content of each image prior to submitting it to the model.  All training images go through the augmentation step; none of the test images are augmented.  Refer to my_classes.py.  The augmentation code is simple:

    def augment_image(self, x):
        val = np.random.randint(0, 4)
        if val == 0:
            x = image.random_brightness(x, (0.01, 0.99))
        elif val == 1:
            x = image.random_rotation(x, 20, row_axis=0, col_axis=1, channel_axis=2)
        elif val == 2:
            x = image.random_shift(x, 0.1, 0.1, row_axis=0, col_axis=1, channel_axis=2) 
        elif val == 3:
            x = image.random_zoom(x, (0.1, 0.1), row_axis=0, col_axis=1, channel_axis=2) 
        elif val == 4:
            x = image.random_shear(x, 30, row_axis=0, col_axis=1, channel_axis=2) 
        else:
            assert False, 'Fatal error, DataGenerator.augment_image(), val out of range'
        return x    

The routine randomly selects one of five image augmentation techniques.  Each one of these techniques performs its work in a random fashion.

Step back from the implementation details and consider the larger picture.  With augmentation enabled, this means that the model will never see the same training image twice.  Effectively, this multiplies the size of our training set and gives us the ability to adequately train a model using a modest dataset. 

### Batch Normalization ###

Dealing with BatchNorm was the most difficult part of this project.  As mentioned above, Transfer learning involves taking an existing pre-trained model, locking down its lower layers (trainable=False), and then training the upper layers with a different dataset.  It turns out that BatchNorm (and Dropout) behave differently when in training mode versus evaluation mode.

Specifically, when in training mode, BatchNorm uses the statistics of the new dataset (fgvc-aircraft-2013b).  This is what we want.  However, when evaluating the performance of our model, BatchNorm uses the statistics of the original dataset (ImageNet).  This is not what we want.

Assume that we've partially trained a model then evaluate its performance. Below is evidence of the BatchNorm issue:

|    |           |  |   Train | |   Eval |
|    |           |  |   ----- | |   ---- |
|    | top-1     |  |   0.639 | |  0.010 |
|    | top-5     |  |   0.831 | |  0.051 |

The train column refers to the top one and five accuracies of the final epoch when running model.fit() using the training set.  The eval column refers to the same accuracies when running model.evaluate() using the test set.  This is puzzling because the train column suggests the model is converging; the eval column strongly suggests otherwise.

There is significant Internet discussion on this issue.  Below is a good link:

    [http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/](URL)

In this post, V. Vryniotis describes the problem, how to reproduce it, an implied workaround and proposes Keras patch to resolve this issue.  So far, his patch has not been incorporated into Keras.

Full disclosure.  I'm still learning Keras and was hesitant to rely on a patch which wasn't accepted.  I chose the workaround instead.  First, a few Keras details.

Internally, Keras runs in either training or test mode.  Normally, the user does not manipulate this attribute; Keras handles this automatically.

When executing:


  * model.fit()
 
Keras is in training mode.

When executing:

* model.evaluate()
* model.predict()
* validation step of model.fit()

Keras is in test mode.

Without a workaround in place, when we test model performance using model.evaluate(), Keras internally switches to test mode, uses the ImageNet statistics in the BatchNorm layers and we have problems.

Keras provides set_learning_phase() for explicitly controlling training versus test mode.  Learning phase must be set prior to creating the associated model.  After creating the model, learning phase is fixed to the value previously set.  I.e., when its value is explicitly set, the methods fit(), evaluate(), predict(), etc. will not change it.

The workaround is to set learning_phase to one prior to creating the model.  

        K.set_learning_phase(1)
        print('  setting learning phase to 1 prior to creating model and loading weights')
 
This locks Keras into training mode.  With this change in place, all three models perform acceptably.

This is just a workaround.  As described V. Vryniotis' post, BatchNorm, when in test mode, uses the mean and variance of the mini-batch (fgvc-aircraft-2013b dataset) for scaling operations.  When in test mode, BatchNorm uses the moving average of the mean and variance (ImageNet dataset).   Intuitively, we would like to use the moving average of mean and variance, using fgvc-aircraft-2013b data, when evaluating the performance of a given model.  The workaround does not give us this capability.




## Execution ##

### Program Structure ###

The overall structure of the tuning code is simple:

    exec_model.py          High level driver file, set key operating parameters
    my_classes.py          Implementation of data generator and data augmentation
    my_utils.py            Code to populate class, partition and labels dictionaries,
                           plus miscellaneous utility routines
    vgg16.py, resent50.py  Child modules of the driver file.
    inceptionv3.py         

As you would expect, the child modules implement the same operations for their respective
Keras applications:

* load pre-trained Keras application (top=False), install top layers, return a Keras Model
* set application specific parameters

The tuning code uses a dictionary, model_params, to organize all parameters into a single structure.  To help with hyperparameter tuning, each program run displays the contents of this dictionary.  For example:

    Model Details
      Name: vgg16 rev-6-a
      Total layers: 25
      Non-trainable layers: 14
      Top Architecture:
        flatten_1	(None, 25088)
        dense_1	(None, 4096)
        dropout_1	(None, 4096)
        dense_2	(None, 4096)
        dropout_2	(None, 4096)
        dense_3	(None, 100)
      Number classes: 100
      Optimizer : sgd
        Learning rate: 0.001
      Image shape: (224, 224, 3)
      Training partitions: ['train', 'val']
      Training images: 6667
      Test partitions: ['test']
      Test images: 3333
      Augmentation: True
      Batch size: 8
      Epochs: 32
      Run mode: train

A typical tuning session involves setting the model, rev and run_mode variables in exec_model.py, then executing the script.

### Run mode ###

The tuning code supports five different modes of operation.

    train          Using training data, train the designated model for the specified number of epochs.
    eval           Using test data, evaluate the performance of the designated model.
    train-eval     Train then evaluate desired model.
    summary        Write a summary of the current model to StdOut.
    noop           No operation, useful for verifying everything ok before training.

### Logs directory ###

The logs directory contains files from previous runs of the tuning code. For example, below are the log files for rev-a of the vgg16 model:

    Training output: vgg16-a-0.txt, vgg16-a-1.txt, vgg16-a-2.txt
    Evaluation output: vgg16-a-eval.txt
    Summary output: vgg16-a-summary.txt
    
### Saved_models directory ###

This directory is empty in the git archive.  It is populated with model weights as part of program execution.  At program start, code determines if an appropriately named weights file exists in this directory.  If so, logic loads the model's weights and uses them for subsequent training or evaluation operations.  Otherwise, code creates the model from scratch. On training completion, logic saves the updated model weights back to this directory.

### Hyperparameters ###

The tuning code supports multiple hyperparameters.  The more important ones are mentioned below.

#### model_params['locked_layers'] ####

Locked layers controls how many of the lower layers of a model are 'locked', that is not trainable.  From machine learning theory, we expect that the lower layers of a model tuned on the ImageNet dataset (our case) would work without alteration for a dataset of aircraft variants.  Locked layers is a key idea in taking a model trained on dataset A, and then fine tuning it to recognize objects in dataset B.  

Setting run_mode='summary' will generate a model summary.  The resulting output is helpful in setting a logical value for locked layers.  See imagenetV3-b-summary.txt in the logs directory for an example.

#### model_params['optimizer_name'] ####

Optimizer name specifies the optimizer used for the training runs.  Only adam and sgd are supported; it is trivial to add others.

#### model_params['learning_rate'] ####

Sets the learning rate for the chosen optimizer.

#### Top Architecture ####

Each model supports two different top architectures.  Resnet50 examples:

    Top D:
      flatten_1	(None, 100352)
      dense_1	(None, 512)
      dropout_1	(None, 512)
      dense_2	(None, 256)
      dropout_2	(None, 256)
      dense_3	(None, 100)

    Top E:
      global_average_pooling2d_1	(None, 2048)
      dense_1	(None, 1024)
      dropout_1	(None, 1024)
      dense_2	(None, 100)

A model's top layers composition strongly impacts performance.  See the Results section.

## Results ##

My goal all along was to achieve accuracy comparable to that published in the Keras documentation for models fine tuned to a different dataset.  Before discussing results, below is the raw data for the various models.

### Vgg16 ###

The Vgg16 models share the following hyperparameters:

* optimizer - stochastic gradient descent
* learing rate - 0.001
* locked layers - 14

Below are the top architectures for the A and B models.

    Top-A:
      flatten_1	(None, 25088)
      dense_1	(None, 4096)
      dropout_1	(None, 4096)
      dense_2	(None, 4096)
      dropout_2	(None, 4096)
      dense_3	(None, 100)

    Top-B:
      global_average_pooling2d_1	(None, 512)
      dense_1	(None, 1024)
      dropout_1	(None, 1024)
      dense_2	(None, 100)


#### Performance ####
   

| Model |  | Total Params |  | Trainable Params |  | Depth |  | Epochs |  | Time |  | Top-1 |  | Top-5 |
| ----- |  | ------------ |  | ---------------- |  | ----- |  | ------ |  | ---- |  | ----- |  | ----- |
|   A   |  |   134.67M    |  |     127.03M      |  |   25  |  |  128   |  | 4:12 |  | 0.724 |  | 0.905 |
|   B   |  |    15.34M    |  |       7.71M      |  |   23  |  |   96   |  | 2:42 |  | 0.753 |  | 0.939 |
| Keras |  |   138.36M    |  |                  |  |   23  |  |        |  |      |  | 0.713 |  | 0.901 |

Where:

* Epochs - number of training epochs to achieve stated accuracy
* Time - training time in hours:minutes to achieve stated accuracy
* Top-1 - Top 1 accuracy of trained model when evaluated using test data
* Top-5 - Top 5 accuracy of trained model when evaluated using test data
* Keras - characteristics of the Keras Vgg16 model

### Resnet50 ###

The Resnet50 models share the following hyperparmeters:

* optimizer - adam
* learning rate - 0.0001
* locked layers - 79

Refer to the Hyperparameters section for the two Resnet50 top architectures.

#### Performance ####

| Model |  | Total Params |  | Trainable Params |  | Depth |  | Epochs |  | Time |  | Top-1 |  | Top-5 |
| ----- |  | ------------ |  | ---------------- |  | ----- |  | ------ |  | ---- |  | ----- |  | ----- |
|   D   |  |    75.13M    |  |      73.62M      |  |  181  |  |  160   |  | 5:00 |  | 0.737 |  | 0.913 |
|   E   |  |    25.79M    |  |      24.29M      |  |  179  |  |   32   |  | 1:00 |  | 0.785 |  | 0.945 |
| Keras |  |    25.64M    |  |                  |  |  168  |  |        |  |      |  | 0.749 |  | 0.921 |


### InceptionV3 ###

The Inception models share the following hyperparameters:

* optimizer - stochastic gradient descent
* learing rate - 0.001
* locked layers - 165

Below are the top architectures for the B and C models.

    Top B:
      flatten_1	(None, 131072)
      dense_1	(None, 512)
      dropout_1	(None, 512)
      dense_2	(None, 256)
      dropout_2	(None, 256)
      dense_3	(None, 100)

    Top C:
      global_average_pooling2d_1	(None, 2048)
      dense_1	(None, 1024)
      dropout_1	(None, 1024)
      dense_2	(None, 100)

#### Performance ####

| Model |  | Total Params |  | Trainable Params |  | Depth |  | Epochs |  | Time  |  | Top-1 |  | Top-5 |
| ----- |  | ------------ |  | ---------------- |  | ----- |  | ------ |  | ----  |  | ----- |  | ----- |
|   B   |  |    89.07M    |  |      83.91M      |  |  317  |  |  272   |  | 15:00 |  | 0.777 |  | 0.927 |
|   C   |  |    24.00M    |  |      18.84M      |  |  315  |  |   32   |  |  1:45 |  | 0.805 |  | 0.951 |
| Keras |  |    23.85M    |  |                  |  |  159  |  |        |  |       |  | 0.779 |  | 0.937 |


