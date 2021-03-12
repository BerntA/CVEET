import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=cfg)

"""
Move raw collected data into the correct hierarchy:

  train
    ---> class1
        ---> class_image_here
    ---> class2
        ---> class_image_here
        
See https://tfhub.dev/s?module-type=image-feature-vector&tf-version=tf2 for diff. pre-trained models.
"""

def generateFigures(mdl, hist, size=(8,6)):
    plt.ioff()
    try:    
        fig = plt.figure(figsize=size)
        plt.ylabel("Loss (training and validation)")
        plt.xlabel("Training Steps")
        plt.ylim([0,2])
        plt.plot(hist["loss"])
        plt.plot(hist["val_loss"])
        plt.tight_layout()
        plt.savefig('../images/temp/plot_{}_1.png'.format(mdl))
        plt.close(fig)

        fig = plt.figure(figsize=size)
        plt.ylabel("Accuracy (training and validation)")
        plt.xlabel("Training Steps")
        plt.ylim([0,1])
        plt.plot(hist["accuracy"])
        plt.plot(hist["val_accuracy"])
        plt.tight_layout()
        plt.savefig('../images/temp/plot_{}_2.png'.format(mdl))
        plt.close(fig)
    except Exception as e:
        print('Error:', e)
    finally:
        plt.show()

def createModel(model_hub_url):
    BATCH_SIZE = 32
    SIZE = (299, 299)

    train = ImageDataGenerator(
        rescale = 1.0/255.0, 
        rotation_range = 40, 
        width_shift_range = 0.2, 
        height_shift_range = 0.2, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True
    )

    test = ImageDataGenerator(
        rescale = 1.0/255.0
    )

    train_gen = train.flow_from_directory(
        "../images/train/", 
        color_mode="rgb", 
        batch_size = BATCH_SIZE, 
        class_mode="categorical", 
        target_size = SIZE, 
        interpolation="bilinear",
        shuffle=False
    )

    test_gen = test.flow_from_directory(
        "../images/test/", 
        color_mode="rgb", 
        batch_size = BATCH_SIZE, 
        class_mode="categorical", 
        target_size = SIZE, 
        interpolation="bilinear",
        shuffle=False
    )

    # Load Pre-Trained Model, and freeze the weights.
    # Define the last layer of the CNN, a fully connected layer.
    print("Creating model...")
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=SIZE+(3,)),
        hub.KerasLayer(model_hub_url, trainable=False),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(
            train_gen.num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )
    ])
    model.build((None,)+SIZE+(3,))
    model.summary()
    model.compile(
        optimizer = RMSprop(lr=0.001), 
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1), 
        metrics = ['accuracy']
    )

    # Train the model on our data.
    print("Training model...")
    hist = model.fit_generator(
        train_gen,
        validation_data = test_gen,
        validation_steps = (test_gen.samples // test_gen.batch_size),
        steps_per_epoch = (train_gen.samples // train_gen.batch_size),    
        epochs = 5
    ).history

    print("Exported figures to ../images/temp.")
    generateFigures('inception', hist)

    print("Saving model...")
    model.save('../exported-models/inception')

if __name__ == "__main__":
    print("TF version:", tf.__version__)
    print("Hub version:", hub.__version__)
    createModel('https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4')
    print("Finished training model!")
