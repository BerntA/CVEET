import os
import numpy as np
import datetime as dt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=cfg)

TARGET_SIZE = (224, 224) # Model specific, see TF hub page for details.
LBLS = sorted(['bike', 'bus', 'car', 'person', 'truck'])

def createEvaluationStatistics(model, test_gen, file):
    try:                
        print("Predicting...")
        idx = 0
        for i in range(test_gen.samples):
            print((i+1), '/', test_gen.samples)
            images, labels = test_gen.next()
            truth = LBLS[np.argmax(labels[0])]
            pred = LBLS[np.argmax(model(images, training=False))]
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
        batch_size = 1, 
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
