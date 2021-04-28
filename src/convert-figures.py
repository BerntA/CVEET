import os
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=cfg)

BATCH_SIZE = 32
TARGET_SIZE = (224, 224) # Model specific, see TF hub page for details.
EXCLUDED = set(['efficientdet_d0', 'ssd_mobilenet_v2', 'ssd_mobilenet_v2_old']) # Excluded folders/mdls.

def createConfusionMatrix(mdl, y, y_pred):
    plt.ioff()
    try:
        a = confusion_matrix(y, y_pred)
        class_names = sorted(['bike', 'bus', 'car', 'person', 'truck'])
        tick_marks = np.arange(len(class_names))

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(a, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        #plt.colorbar()
        plt.xticks(tick_marks, class_names, rotation=0)
        plt.yticks(tick_marks, class_names, rotation=90)

        for r in range(len(class_names)):
            for c in range(len(class_names)):
                plt.text(c, r, '{} ({:.3f})'.format(a[r,c], (a[r,c]/a[r,:].sum())), horizontalalignment="center", color='green')
                
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('../images/temp/plot_{}_3.png'.format(mdl))
        plt.close(fig)
    except Exception as e:
        print('Error:', e)
    finally:
        plt.show()

def loadModelAndConvertFig(mdlname, test_gen):
    try:
        print("Loading Model ", mdlname, '...')
        model = tf.keras.models.load_model('../exported-models/{}'.format(mdlname))
        print("Predicting on eval. data...")
        probs = model.predict(test_gen)
        probs = np.argmax(probs, axis=1).flatten() # Retain the indicies for the highest prob. classes.
        print("Exported confusion matrix to ../images/temp.")
        createConfusionMatrix(mdlname, test_gen.classes.copy(), probs)
    except Exception as e:
        print('Error:', e)

if __name__ == "__main__":
    print("TF version:", tf.__version__)
    listGPUs = tf.config.list_physical_devices('GPU')
    if len(listGPUs) == 0:
        print("No GPUs detected, script might be slow!")
    print("Loading Test Data...")

    test = ImageDataGenerator(
        rescale = 1.0/255.0
    )

    test_gen = test.flow_from_directory(
        "../images/test/",
        color_mode="rgb", 
        batch_size = BATCH_SIZE, 
        class_mode="categorical", 
        target_size = TARGET_SIZE, 
        interpolation="bilinear",
        shuffle=False
    )

    for f in os.listdir('../exported-models/'):
        f = f.strip().lower()
        if f in EXCLUDED:
            continue
        loadModelAndConvertFig(f, test_gen)
