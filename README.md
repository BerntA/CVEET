# CVEET
Classifying Vehicles Entering and Exiting a tunnel, MSc thesis.

## General Overview
* annotations - Object Detection Train & Test binary files + Label files.
* exported-models - Saved Models.
* fig - Saved Figures, Plots, etc.
* images - Image data and annotation data.
* logs - Various logs for profiling, evaluation, etc.
* models - Object Detection model configurations.
* pre-trained-models - Object Detection pre-trained models.
* report - Presentation and pdf report.
* src - Source Code.

## Code Overview
* preprocessing.ipynb - Used for splitting the data into train and test, creates a file which holds the filenames rather than copying over the same files. Additionally it includes some histogram plots of the various vehicle distributions, and an algorithm for extracting the objects in the annotated images, such that they can be used in a regular CNN.
* gen-tf-record.ipynb - Generates the TF Record files from the train and test image folders. These folders contain the mapping files which hold images and their respective annotation file paths.
* convert-data-labels.ipynb - Is used to convert other annotation formats, and image formats to the correct formats. Various parsers and translation stuff logic is found here.
* webcam-scraper [.ipynb, .py] - Is used for fetching images from certain webcams.
* utils.py - Utility functions for plotting and parsing XML.
* stream.py - Realtime R-CNN / SSD based object detection system, utilizes PyQt5 for the GUI.
* object-detection.ipynb - Used for testing the R-CNN / SSD approach on single images.
* object-detection-bgsub.ipynb - Used for testing vanilla CNN with BG subtraction logic.
* detection.py - Contains object detection utility functions for the R-CNN / SSD approach.
* eval-stat.py - Used for creating plots for the evaluation of the custom data generator.
* transfer-learning.py - Used for training vanilla CNN, doing hyperparam tuning, testing distributions, etc.
* testing.ipynb - Misc testing, and various plotting.
* profiling.py - Used for profiling the CNN and region proposal method.

## Executing Training
Run train.sh for object-detection or train-cnn for a general CNN. Export object-detection model with exportmdl.sh.

## Prerequisites (pip packages)
* OpenCV
* Tensorflow
* Tensorflow Hub
* Tensorflow Object Detection API
* NumPy
* matplotlib
* scipy
* sklearn
* PyQt5
* nvidia_smi
* psutil

## Scoreboard
| Feature Vector  | Accuracy | Epochs | Batch Size | Optimizer | Learning Rate | Dropout | Label Smoothing | Regularization | Generator |
|:---------------:|:-----:|:--:|:--:|:----:|:-----:|:---:|:---:|:--------:|:------:|
|    MobileNET + Tuning    | 0.949 | 30 | 64 | RMSProp | 0.001 | 0.1 | 0.15 | L1 0.001 | Custom - Distr [0.60, 0.70, 0.25, 0.35, 0.65] |
|    EfficientNET b7    | 0.930 | 50 | 32 | RMSProp | 0.001 | 0.25 | 0.15 | L1 0.001 | Custom |
|    MobileNET    | 0.927 | 50 | 32 | Adam | 0.001 | 0.2 | 0.2 | L1 0.001 | Custom |
|    MobileNET    | 0.922 | 65 | 32 | RMSProp | 0.001 | 0.2 | 0.15 | L1 0.001 | Custom |
|    MobileNET    | 0.900 | 65 | 32 | Adam | 0.01 | 0.3 | 0.2 | L1 0.001 | Custom |
|    MobileNET    | 0.940 | 100 | 16 | SGD | 0.001 | 0.1 | 0.1 | L1 0.001 | Less Data |
|    MobileNET    | 0.810 | 80 | 16 | SGD | 0.001 | 0.0 | 0.0 | L1 0.001 | Default |
|    InceptionNET    | 0.760 | 4 | 16 | RMSProp | 0.0001 | 0.1 | 0.1 | L2 0.0001 | Default |

</br>

## R-CNN Realtime Demo
[![Alt text](https://img.youtube.com/vi/zNQiT8J-XMI/0.jpg)](https://www.youtube.com/watch?v=zNQiT8J-XMI)

## Data
* [CVEET Data](https://drive.google.com/file/d/1sySRD7BlavonDOqozSYw7PNcW08ERojS/view?usp=sharing)
* [KITTI Data](https://www.kaggle.com/twaldo/kitti-object-detection) 
* [ExDark Data](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset) 

## Misc
* [CVEET Exported Models & Pre-Trained Models](https://drive.google.com/file/d/16Pe13B1kOSlBPp3HtDQZPhE4RfGJJxgg/view?usp=sharing)
* Check the report folder for a presentation of the work and pdf of the thesis.
