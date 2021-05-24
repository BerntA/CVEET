import os
import sys
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import tensorflow as tf

cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=cfg)

EXCLUDED = set(['efficientdet_d0', 'ssd_mobilenet_v2', 'ssd_mobilenet_v2_old']) # Excluded folders/mdls.

if __name__ == "__main__":
    print("TF version:", tf.__version__, '\n')
    with open('../logs/models.txt', 'w') as wr:
        for f in os.listdir('../exported-models/'):
            f = f.strip().lower()
            if f in EXCLUDED:
                continue
            model = tf.keras.models.load_model('../exported-models/{}'.format(f))
            print('Model:', f)
            wr.write('Model: {}\n'.format(f))
            model.summary(print_fn=lambda x: wr.write('{}\n'.format(x)))
            wr.write('\n')
    print("Finished!")
    