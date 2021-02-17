"""Object Detection Stream"""

import sys
import os
import cv2
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QLabel, QSizePolicy, QSlider
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QObject, Qt
from PyQt5.QtGui import QImage, QPixmap
#from PyQt5.uic import loadUi

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

#STREAM = 'https://kamera.vegvesen.no/public/0329001_1/hls_1_stream_1_orig.m3u8'
STREAM = 'https://kamera.vegvesen.no/public/1129024_1/hls_1_stream_1_orig.m3u8'
#DETMODEL = '../exported-models/efficientdet_d0/saved_model'
DETMODEL = '../exported-models/ssd_mobilenet_v2/saved_model'
THRESHOLD = 0.4

class ObjectDetector(QObject):
    pixmap = pyqtSignal(object)

    def __init__(self):
        """Initialize worker thread."""
        super().__init__()
        self.shutdown = False
        self.vstream = None
        self.detect_fn = None
        self.category_index = None
        self.thread = None

    def open(self):
        """Start the worker."""
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self.run)
        self.thread.start()

    def close(self):
        """Shutdown the worker."""
        self.shutdown = True
        while self.thread:
            time.sleep(0.05)
        time.sleep(0.5)
        print("Shutdown worker thread gracefully!")

    def inference(self, image_np):
        """Run inference on an image using our exported model."""
        global THRESHOLD
        try:
            input_tensor = tf.convert_to_tensor(image_np)
            input_tensor = input_tensor[tf.newaxis, ...]
            detections = self.detect_fn(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=100,
                min_score_thresh=THRESHOLD,
                agnostic_mode=False,
                line_thickness=1,
                mask_alpha=0.4
            )            
        except Exception as error:
            print('Error:', error)
        finally:
            return image_np

    def run(self):
        """Run Real Time Obj. Detection on a camera stream."""
        global STREAM, DETMODEL
        print("Loading detection function...")
        self.category_index = label_map_util.create_category_index_from_labelmap('../annotations/label_map.pbtxt', use_display_name=True)
        self.detect_fn = tf.saved_model.load(DETMODEL)
        try:
            print("Started Video Stream")
            self.vstream = cv2.VideoCapture(STREAM)
            while (not self.shutdown):
                ret, frame = self.vstream.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame[25:,:,:] # Crop away the top black bar.
                    #frame = cv2.resize(frame, (512, 512)) # Some models need a power of two img for inference...
                    frame = self.inference(frame)
                    frame = QPixmap.fromImage(QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888))
                    self.pixmap.emit(frame)
                time.sleep(0.5)
        except Exception as e:
            print("Worker Error:", e)
        finally:
            self.vstream.release()
            self.vstream = None
            self.detect_fn = None
            self.category_index = None
            self.thread = None
            print("Ended the Video Stream Thread")

class MainWindow(QMainWindow):
    def __init__(self):
        """Initialize the window."""
        super().__init__()
        self.pixmap = None
        self.setupUi()
        self.show()

        self.worker = ObjectDetector()
        self.worker.pixmap.connect(self.receiveFrame)
        self.worker.open()

    def setupUi(self):
        """Load/Create UI controls."""
        self.setFixedWidth(640)
        self.setFixedHeight(480)        

        self.pixmap = QLabel(self)
        self.pixmap.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.pixmap.setScaledContents(True)
        self.pixmap.resize(self.width(), self.height()-20)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setGeometry(0, self.height()-20, self.width(), 20)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(40)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.thresholdChanged)

        self.setWindowTitle("Video Stream - Object Detection ({}%)".format(self.slider.value()))

    def closeEvent(self, event):
        """Handle graceful shutdown."""
        self.worker.close()
        self.worker = None
        super().closeEvent(event)

    def thresholdChanged(self):
        """Slider value changed."""
        global THRESHOLD
        THRESHOLD = (float(self.slider.value()) / 100.0)
        self.setWindowTitle("Video Stream - Object Detection ({}%)".format(self.slider.value()))

    @pyqtSlot(dict)
    def receiveFrame(self, frame):
        """Received a frame from the worker thread. (img)"""
        self.pixmap.setPixmap(frame)

if __name__ == "__main__":
    """App Entry Point."""
    print("Using TF version:", tf.__version__)
    app = QApplication(sys.argv)
    mainwnd = MainWindow()
    sys.exit(app.exec())
