import os
import sys
import time
import numpy as np
import datetime as dt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=cfg)

BATCH_SIZE = 32
TARGET_SIZE = (224, 224) # Model specific, see TF hub page for details.
LBLS = sorted(['bike', 'bus', 'car', 'person', 'truck'])

def createEvaluationStatistics(model, test_gen, file):
    try:                
        print("Predicting...")
        idx = 0
        for i in range((test_gen.samples // BATCH_SIZE)):
            print((i+1), '/', (test_gen.samples // BATCH_SIZE))
            images, labels = test_gen.next()
            for img, lbl in zip(images, labels):
                truth = LBLS[np.argmax(lbl)]
                pred = LBLS[np.argmax(model(np.expand_dims(img, axis=0), training=False))]
                filename = test_gen.filenames[idx]
                idx += 1
                if truth == pred:
                    continue
                file.write('{},{},{}\n'.format(truth, pred, filename))
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

    with open('../logs/eval-{}.txt'.format(dt.datetime.now().strftime("%d-%m-%H-%M")), 'w') as f:
        print("Loading Model...")
        model = tf.keras.models.load_model('../exported-models/mobilenet_new_5')
        createEvaluationStatistics(model, test_gen, f)
        print("Finished, wrote results to logs!")
