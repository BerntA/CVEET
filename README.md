# CVEET
Classifying Vehicles Entering and Exiting a tunnel, MSc thesis.

## Experiments

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
