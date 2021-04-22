# CVEET
Classifying Vehicles Entering and Exiting a tunnel, MSc thesis.

## Experiments

| Feature Vector  | Accuracy | Epochs | Batch Size | Optimizer | Learning Rate | Dropout | Label Smoothing | Regularization | Generator |
|:---------------:|:-----:|:--:|:--:|:----:|:-----:|:---:|:---:|:--------:|:------:|
|    MobileNET    | 0.927 | 32 | 50 | Adam | 0.001 | 0.2 | 0.2 | L1 0.001 | Custom |
|    MobileNET    | 0.922 | 32 | 65 | RMSProp | 0.001 | 0.2 | 0.15 | L1 0.001 | Custom |
|    MobileNET    | 0.900 | 32 | 65 | Adam | 0.01 | 0.3 | 0.2 | L1 0.001 | Custom |
|    MobileNET    | 0.940 | 16 | 100 | SGD | 0.001 | 0.1 | 0.1 | L1 0.001 | Less Data |
|    MobileNET    | 0.810 | 16 | 80 | SGD | 0.001 | 0.0 | 0.0 | L1 0.001 | Default |
|    InceptionNET    | 0.760 | 16 | 4 | RMSProp | 0.0001 | 0.1 | 0.1 | L2 0.0001 | Default |

</br>