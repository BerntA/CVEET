import os
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.regularizers import l1, l2
from sklearn.metrics import confusion_matrix
from scipy.stats import bernoulli

cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=cfg)

HUB_URL = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5" # URL for feat. vec. (pre-trained mdl)
BATCH_SIZE = 64
TARGET_SIZE = (224, 224) # Model specific, see TF hub page for details.

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

class BalancedDataGenerator(tf.keras.utils.Sequence):
    """
    Give some classes less probability to be picked in a batch than others.
    Revert to pure random if none gets chosen.
    """
    def __init__(
        self,
        imgDataGen,
        distribution, # Determines the prob. here! (bike, bus, car, person, truck)
        target_size=(224, 224), 
        batch_size=16,
        target_path="../images/train/"        
    ):
        self.batches = batch_size
        self.classes = sorted(['car', 'truck', 'bus', 'person', 'bike'])
        self.aug = imgDataGen
        self.generators = []
        self.num_samples = 0
        for c in self.classes:
            gen = self.aug.flow_from_directory(
                target_path, 
                color_mode="rgb", 
                batch_size=1, 
                class_mode="categorical", 
                target_size=target_size, 
                interpolation="bilinear",
                shuffle=True,
                classes=[c]
            )
            self.generators.append(gen)
            self.num_samples += gen.samples        
        self.length = (self.num_samples // self.batches)
        self.distr = np.array(distribution)
        self.distr_order = np.argsort(distribution)
        self.on_epoch_end()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        X, y = [], np.zeros((self.batches, len(self.classes)), dtype=np.float32)
        for b in range(self.batches):
            i = -1
            for idx in self.distr_order:                
                if bernoulli.rvs(p=self.distr[idx]):
                    i = idx                    
                    break
            
            if i == -1: # If we failed to select any class above, pick random. 50-50...
                i = np.random.randint(len(self.classes))                
                    
            img, _ = self.generators[i].next()
            y[b,i] = 1.0
            X.append(img)

        return np.concatenate(X), y

    def on_epoch_end(self):
        pass # We do nothing, our generators does the 'hard' work.

def createModel(mdlname, batch_size, epochs=20, optimizer=RMSprop, learn_rate=0.001, dropout=0.25, label_smoothing=0.15, regularizer=l1, regularizer_value=0.001, traintune=False, gridsearch=False, logTensorBoard=False, createFigures=True, generatorDistribution=[0.55, 0.85, 0.30, 0.40, 0.75]):
    train = ImageDataGenerator(
        rescale = 1.0/255.0, 
        rotation_range = 30, 
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
        batch_size = batch_size, 
        class_mode="categorical", 
        target_size = TARGET_SIZE, 
        interpolation="bilinear",
        shuffle=True
    )

    test_gen = test.flow_from_directory(
        "../images/test/", 
        color_mode="rgb", 
        batch_size = batch_size, 
        class_mode="categorical", 
        target_size = TARGET_SIZE, 
        interpolation="bilinear",
        shuffle=False
    )

    new_train_gen = BalancedDataGenerator(
        train,
        generatorDistribution,
        target_size=TARGET_SIZE,
        batch_size=batch_size,
        target_path="../images/train/"
    )

    # Load Pre-Trained Model, and freeze the weights.
    # Define the last layer of the CNN, a fully connected layer.
    if not gridsearch:
        print("Creating model...")

    model = tf.keras.Sequential([
        InputLayer(input_shape=TARGET_SIZE+(3,)),
        hub.KerasLayer(HUB_URL, trainable=traintune),
        Dropout(rate=dropout),
        Dense(1024, activation='relu'),
        Dropout(rate=dropout),
        Dense(train_gen.num_classes, kernel_regularizer=regularizer(regularizer_value), activation='softmax')
    ])
    model.build((None,)+TARGET_SIZE+(3,))
    if not gridsearch:
        model.summary()
    model.compile(
        optimizer = optimizer(learning_rate=learn_rate), 
        loss = CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing), 
        metrics = ['accuracy']
    )

    # Train the model on our data.
    if not gridsearch:
        print("Training model...")

    hist = model.fit(
        new_train_gen,
        validation_data = test_gen,
        validation_steps = (test_gen.samples // test_gen.batch_size),
        steps_per_epoch = (train_gen.samples // train_gen.batch_size),    
        epochs = epochs,
        shuffle = False,
        callbacks=([tf.keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=1)] if logTensorBoard else None)
    ).history

    if not gridsearch:
        print("Evaluating model...")

    loss, acc = model.evaluate(test_gen)
    if not gridsearch:
        print(loss, acc)

    if createFigures:
        if not gridsearch:
            print("Predicting on eval. data...")
        probs = model.predict(test_gen)
        probs = np.argmax(probs, axis=1).flatten() # Retain the indicies for the highest prob. classes.
        if not gridsearch:
            print("Exported figures to ../images/temp.")
        generateFigures(mdlname, hist, test_gen.classes.copy(), probs)

    if gridsearch:
        return loss, acc
    else:
        print("Saving model...")
        model.save('../exported-models/{}'.format(mdlname))

def gridSearchOptimize(mdlname, optimizeType, verbose=True):
    if not os.path.exists('../models/{}'.format(mdlname)):
        os.mkdir('../models/{}'.format(mdlname))

    K.clear_session() # Clear any previous session!
    iteration = 1
    res = []
    start_time, iteration_time = time.time(), time.time()

    if optimizeType == 'generator': # Test various distributions for the new data generator.
        distr = [
            [0, 0, 0, 0, 0], # random 50%
            [0.50, 0.80, 0.25, 0.35, 0.75],
            [0.60, 0.70, 0.25, 0.35, 0.65],
            [0.70, 0.60, 0.25, 0.35, 0.55],
            [0.80, 0.50, 0.25, 0.35, 0.45],
            [0.40, 0.90, 0.25, 0.35, 0.85],
            [0.40, 0.80, 0.25, 0.35, 0.90],
            [0.40, 0.90, 0.25, 0.35, 0.90],
            [0.50, 0.80, 0.55, 0.45, 0.75],
            [0.60, 0.80, 0.65, 0.55, 0.75],
            [0.60, 0.70, 0.75, 0.65, 0.60]
        ]
        print("Started generator distr. tuning...")
        for i,d in enumerate(distr):
            loss, acc = createModel('{}_{}'.format(mdlname, (i+1)), BATCH_SIZE, gridsearch=True, generatorDistribution=d)
            K.clear_session() # Clear memory used. Proceed to next!
            res.append([loss, acc] + d)
            print('{} / {}'.format((i+1), len(distr)))
        print("Finished generator distribution tuning, time elapsed: {:.4f} sec.".format((time.time() - start_time)))
    else: # Hyperparam. tuning (slow)
        batch_sizes = [16]
        optimizers = [RMSprop, Adam, SGD]
        learn_rates = [0.001, 0.01, 0.1]    
        dropouts = [0, 0.1, 0.2]
        label_smoothening = [0, 0.1, 0.2]
        regularizers = [l1, l2]
        regularizer_values = [0.0001, 0.001]
        allow_training = [False]
        print("Started hyperparameter tuning...")
        for batchSize in batch_sizes:
            for optimizer in optimizers:
                for learnRate in learn_rates:
                    for dropOutRate in dropouts:
                        for lblSmooth in label_smoothening:
                            for reg in regularizers:
                                for regVal in regularizer_values:
                                    for allowTuning in allow_training:
                                        loop_time = time.time()
                                        loss, acc = createModel(mdlname, batchSize, 3, optimizer, learnRate, dropOutRate, lblSmooth, reg, regVal, allowTuning, True, createFigures=False)
                                        res.append([
                                            loss, acc, batchSize, optimizer, learnRate, dropOutRate, lblSmooth, reg, regVal, allowTuning
                                        ])
                                        if verbose and ((iteration % 8) == 0):
                                            print(iteration, "Processed ---> {:.3f} LOSS, {:.3f} ACCU (this {:.4f} sec, overall {:.4f} sec)".format(loss, acc, (time.time() - loop_time), (time.time() - iteration_time)))
                                            iteration_time = time.time()
                                        iteration += 1
                                        K.clear_session() # Clear memory used. Proceed to next!
        print("Finished hyperparameter tuning, time elapsed: {:.4f} sec.".format((time.time() - start_time)))
    
    K.clear_session()
    if len(res) == 0:
        print("No results were recorded?!")
        return

    res.sort(key = lambda x: x[1], reverse=True) # Sort on ACC, high->low
    print("Writing results to ../models/{}/stat.csv".format(mdlname))
    with open('../models/{}/stat.csv'.format(mdlname), 'w') as f:
        for obj in res:
            s = ','.join([str(v) for v in obj])
            f.write('{}\n'.format(s))

    print("Creating best model based on highest acc...")    
    mdl = res[0][2:]
    if optimizeType == 'generator':
        createModel(mdlname, BATCH_SIZE, 30, generatorDistribution=mdl)
    else:
        createModel(mdlname, mdl[0], 30, mdl[1], mdl[2], mdl[3], mdl[4], mdl[5], mdl[6], mdl[7])
    K.clear_session()

if __name__ == "__main__":
    print("TF version:", tf.__version__)
    print("Hub version:", hub.__version__)
    listGPUs = tf.config.list_physical_devices('GPU')
    if len(listGPUs) == 0:
        print("No GPUs available, terminating!")
    else:
        if len(sys.argv) > 1: # Start grid search logic...
            print("Starting tuning procedure...")
            gridSearchOptimize('mobilenet_gen_opt', 'generator')
        else:
            createModel('mobilenet_new_7', BATCH_SIZE, logTensorBoard=True)
        print("Finished training model!")
        