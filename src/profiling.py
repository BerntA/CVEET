import os
import sys
import time
import psutil
import nvidia_smi
import cv2
import numpy as np

from collections import defaultdict
from detection import inference

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from object_detection.utils import label_map_util

cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=cfg)

BATCH_SIZE = 32
TARGET_SIZE_1 = (224, 224)
TARGET_SIZE_2 = (500, 500)
RESULTS = defaultdict(list)

def getMemoryUsage():
    """Returns CPU + GPU Memory Usage"""
    cpu_mem, gpu_mem = 0, 0
    try:
        cpu_mem = psutil.Process(os.getpid()).memory_info().rss
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(int(os.environ['CUDA_VISIBLE_DEVICES']))
        gpu_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
    except Exception as e:
        print("Could not retrieve memory usage:", e)
    finally:
        nvidia_smi.nvmlShutdown()
        return cpu_mem, gpu_mem

def profileRegionProposalLogic():
    print("Profiling Region Proposal Network...")
    K.clear_session() # Clear any previous session!
    category_index = label_map_util.create_category_index_from_labelmap('../annotations/label_map.pbtxt', use_display_name=True)
    models = ['ssd_mobilenet_v2', 'efficientdet_d0']
    test_dataset = tf.data.TFRecordDataset('../annotations/test.record').map(
        lambda x: tf.io.parse_single_example(x, {'image/encoded': tf.io.FixedLenFeature([], tf.string)})
    )
    cpu_start, gpu_start = getMemoryUsage()

    for m in models:
        start_mem = getMemoryUsage()[0]
        start = time.time()        
        mdl = tf.saved_model.load('../exported-models/{}/saved_model'.format(m))
        RESULTS[m].append(time.time()-start)
        gpu_curr = getMemoryUsage()[1]

        total_time, num_items = 0, 0
        for f in test_dataset:
            I = tf.image.decode_jpeg(f['image/encoded'], channels=3).numpy()
            I = cv2.resize(I, (500, 500)) # For efficient net.
            start = time.time()
            _ = inference(mdl, I, tf.convert_to_tensor(I), category_index, False, 0.35)
            total_time += (time.time() - start)
            num_items += 1
            #break

        RESULTS[m].append(total_time)
        RESULTS[m].append(total_time / max(num_items, 1))

        cpu_now, gpu_now = getMemoryUsage()
        gpu_curr = max(gpu_curr, gpu_now)
        RESULTS[m].append((cpu_now - start_mem)/(1024**2))
        RESULTS[m].append((gpu_curr - gpu_start)/(1024**2))

        K.clear_session() # Clear any previous session!

    for v in RESULTS.values():
        print(v)

def profileVanillaCNN():
    print("Profiling Vanilla CNN...")
    K.clear_session() # Clear any previous session!

    test = ImageDataGenerator(
        rescale = 1.0/255.0
    )

    models = [
        ('mobilenet_new_5', test.flow_from_directory("../images/test/", color_mode="rgb", batch_size = BATCH_SIZE, class_mode="categorical", target_size = TARGET_SIZE_1, interpolation="bilinear", shuffle=False)), 
        ('mobilenet_traintune_1', test.flow_from_directory("../images/test/", color_mode="rgb", batch_size = BATCH_SIZE, class_mode="categorical", target_size = TARGET_SIZE_1, interpolation="bilinear", shuffle=False)), 
        ('efficientnet_1', test.flow_from_directory("../images/test/", color_mode="rgb", batch_size = BATCH_SIZE, class_mode="categorical", target_size = TARGET_SIZE_2, interpolation="bilinear", shuffle=False))
    ]
    cpu_start, gpu_start = getMemoryUsage()

    for m, gen in models:
        start_mem = getMemoryUsage()[0]
        start = time.time()

        mdl = tf.keras.models.load_model('../exported-models/{}'.format(m))
        RESULTS[m].append(time.time()-start)
        gpu_curr = getMemoryUsage()[1]
        
        v = (gen.samples // BATCH_SIZE)
        start = time.time()
        probs = mdl.predict(gen)
        t = (time.time()-start)
        RESULTS[m].append(t)
        RESULTS[m].append(t / v)

        cpu_now, gpu_now = getMemoryUsage()
        gpu_curr = max(gpu_curr, gpu_now)
        RESULTS[m].append((cpu_now - start_mem)/(1024**2))
        RESULTS[m].append((gpu_curr - gpu_start)/(1024**2))

        K.clear_session() # Clear any previous session!

    for v in RESULTS.values():
        print(v)

if __name__ == "__main__":
    print("TF version:", tf.__version__)
    listGPUs = tf.config.list_physical_devices('GPU')
    if len(listGPUs) == 0:
        print("No GPUs detected, script might be slow!")
    print("Profiling models...")

    #profileVanillaCNN()
    profileRegionProposalLogic()

    print("\nFinished profiling!")
