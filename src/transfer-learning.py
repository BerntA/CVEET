import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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

def generateFigures(mdl, hist, y, y_pred, size=(8,6)):
    plt.ioff()
    try:
        # Loss Curve
        fig = plt.figure(figsize=size)
        plt.ylabel("Loss")
        plt.xlabel("Training Steps")
        plt.ylim([0,2])
        plt.plot(hist["loss"], label="train")
        plt.plot(hist["val_loss"], label="validation")
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('../images/temp/plot_{}_1.png'.format(mdl))
        plt.close(fig)

        # Accuracy Curve
        fig = plt.figure(figsize=size)
        plt.ylabel("Accuracy")
        plt.xlabel("Training Steps")
        plt.ylim([0,1])
        plt.plot(hist["accuracy"], label="train")
        plt.plot(hist["val_accuracy"], label="validation")
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('../images/temp/plot_{}_2.png'.format(mdl))
        plt.close(fig)

        # Confusion Matrix
        a = confusion_matrix(y, y_pred)
        class_names = ['bike', 'bus', 'car', 'person', 'truck']
        tick_marks = np.arange(len(class_names))

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(a, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xticks(tick_marks, class_names, rotation=0)
        plt.yticks(tick_marks, class_names, rotation=90)

        for r in range(len(class_names)):
            for c in range(len(class_names)):
                plt.text(c, r, a[r, c], horizontalalignment="center", color='green')
                
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('../images/temp/plot_{}_3.png'.format(mdl))
        plt.close(fig)
    except Exception as e:
        print('Error:', e)
    finally:
        plt.show()

def createModel(mdlname, model_hub_url, BATCH_SIZE, SIZE):
    train = ImageDataGenerator(
        rescale = 1.0/255.0, 
        rotation_range = 40, 
        width_shift_range = 0.2, 
        height_shift_range = 0.2, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True,
        brightness_range = [0.2, 1.0]
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
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            activation='softmax'
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
    hist = model.fit(
        train_gen,
        validation_data = test_gen,
        validation_steps = (test_gen.samples // test_gen.batch_size),
        steps_per_epoch = (train_gen.samples // train_gen.batch_size),    
        epochs = 5,
        shuffle = False
    ).history

    print("Evaluating model...")
    loss, acc = model.evaluate(test_gen)
    print(loss, acc)

    print("Predicting on eval. data...")
    probs = model.predict(test_gen)
    probs = np.argmax(probs, axis=1).flatten() # Retain the indicies for the highest prob. classes.

    print("Exported figures to ../images/temp.")
    generateFigures(mdlname, hist, test_gen.classes.copy(), probs)

    print("Saving model...")
    model.save('../exported-models/{}'.format(mdlname))

if __name__ == "__main__":
    print("TF version:", tf.__version__)
    print("Hub version:", hub.__version__)
    listGPUs = tf.config.list_physical_devices('GPU')
    if len(listGPUs) == 0:
        print("No GPUs available, terminating!")
    else:
        createModel(
            'mobilenet', # Name of this mdl
            'https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5', # URL for feat. vec. (pre-trained mdl)
            16, # Batch Size
            (224, 224) # Target Size
        )
        print("Finished training model!")
        