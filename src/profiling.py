import os
import sys
import time
import psutil
import nvidia_smi
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=cfg)

BATCH_SIZE = 32
TARGET_SIZE_1 = (224, 224)
TARGET_SIZE_2 = (500, 500)

if __name__ == "__main__":
    print("TF version:", tf.__version__)
    listGPUs = tf.config.list_physical_devices('GPU')
    if len(listGPUs) == 0:
        print("No GPUs detected, script might be slow!")
    print("Profiling models...")

    test = ImageDataGenerator(
        rescale = 1.0/255.0
    )

    models = [
        ('mobilenet_new_5', test.flow_from_directory("../images/test/", color_mode="rgb", batch_size = BATCH_SIZE, class_mode="categorical", target_size = TARGET_SIZE_1, interpolation="bilinear", shuffle=False)), 
        ('mobilenet_traintune_1', test.flow_from_directory("../images/test/", color_mode="rgb", batch_size = BATCH_SIZE, class_mode="categorical", target_size = TARGET_SIZE_1, interpolation="bilinear", shuffle=False)), 
        ('efficientnet_1', test.flow_from_directory("../images/test/", color_mode="rgb", batch_size = BATCH_SIZE, class_mode="categorical", target_size = TARGET_SIZE_2, interpolation="bilinear", shuffle=False))
    ]
    meas = [[] for _ in models]

    process = psutil.Process(os.getpid())
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    gpu_start = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used

    try:
        for i, (m, gen) in enumerate(models):
            start = time.time()
            start_mem = process.memory_info().rss

            mdl = tf.keras.models.load_model('../exported-models/{}'.format(m))
            gpu_curr = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
            meas[i].append(time.time()-start)
            
            v = (gen.samples // BATCH_SIZE)
            start = time.time()
            probs = mdl.predict(gen)
            t = (time.time()-start)
            meas[i].append(t)
            meas[i].append(t / v)

            gpu_curr = max(gpu_curr, nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used)
            meas[i].append((process.memory_info().rss - start_mem)/(1024**2))
            meas[i].append((gpu_curr - gpu_start)/(1024**2))

            K.clear_session() # Clear any previous session!
    except Exception as e:
        print(e)
    finally:
        nvidia_smi.nvmlShutdown()

    K.clear_session() # Clear any previous session!

    for v in meas:
        print(v)

    print("\nFinished profiling!\n")
