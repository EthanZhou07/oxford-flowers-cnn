# Oxford 102 Flowers Classification with CNN



This project implements a convolutional neural network (CNN) in PyTorch for image classification on the Oxford 102 Flowers dataset. The project includes data preprocessing, model training, validation, test evaluation, and visualization of training results and sample predictions.



## Project Goal



The goal of this project is to build a custom CNN baseline for multi-class flower image classification and analyze its performance through training curves and prediction examples.



## Dataset



This project uses the Oxford 102 Flowers dataset.



\- Number of classes: 102

\- Task: multi-class image classification

\- Data split:

&#x20; - Training set: 70%

&#x20; - Validation set: 15%

&#x20; - Test set: 15%



The dataset is automatically downloaded in the script.



## Model and Method



The model is a custom CNN built in PyTorch. It includes:



\- 4 convolutional layers

\- ReLU activations

\- max pooling layers

\- fully connected classifier

\- dropout regularization



The training pipeline includes:



\- image preprocessing and normalization

\- data augmentation on the training set

\- model training with Adam optimizer

\- validation-based checkpoint saving

\- final evaluation on the test set



## Training Configuration



\- Framework: PyTorch

\- Optimizer: Adam

\- Loss function: CrossEntropyLoss

\- Epochs: 10

\- Batch size: 32

\- Learning rate: 5e-4

\- Weight decay: 5e-4

\- Dropout: 0.5

\- Random seed: 42



## Results



The model was successfully trained on the Oxford 102 Flowers dataset and showed stable improvement during training.



Final performance of the best saved model:



\- Best validation accuracy: 53.91%

\- Test loss: 1.7108

\- Test accuracy: 54.92%



### Loss Curve

![Loss Curve](outputs/figures/loss\_curve.png)



### Accuracy Curve

![Accuracy Curve](outputs/figures/accuracy\_curve.png)



### Sample Predictions

![Prediction Examples](outputs/figures/prediction\_examples.png)


## Baseline Comparison

To evaluate the effectiveness of the custom CNN, compared it with a transfer learning baseline based on **pretrained ResNet18**.

### Experimental Setup

Both models were evaluated on the same Oxford 102 Flowers dataset split:

- Training set: 5732 images
- Validation set: 1228 images
- Test set: 1229 images

The custom CNN was trained from scratch, while the ResNet18 baseline used ImageNet pretrained weights, froze the backbone, and trained only the final fully connected layer.

### Results

| Model | Training Strategy | Best Val Acc | Test Acc | Test Loss |
|---|---|---:|---:|---:|
| Custom CNN | trained from scratch | 53.91% | 54.92% | 1.7108 |
| ResNet18 (pretrained) | frozen backbone + train final fc layer only | 91.29% | 92.35% | 0.3885 |

### Analysis

The pretrained ResNet18 baseline significantly outperformed the custom CNN on this task.

- The custom CNN achieved about **54.92%** test accuracy.
- The pretrained ResNet18 achieved **92.35%** test accuracy.
- This is an improvement of about **37.43%** in test accuracy.

This result suggests that transfer learning is much more effective than training a small CNN from scratch for the Oxford 102 Flowers dataset. A likely reason is that the pretrained ResNet18 already learned strong general visual features from ImageNet, which transfer well to fine-grained flower classification. In contrast, the custom CNN had to learn all feature representations from scratch with a relatively limited dataset.

### Value of the Custom CNN

- Built a complete image classification pipeline in PyTorch from scratch.

- Included data preprocessing, dataset splitting, and image normalization.

- Performed training, validation, and testing on the real-world Oxford 102 Flowers dataset.

- Visualized loss curves, accuracy curves, and prediction results.

- Provided a clear baseline for comparison with pretrained ResNet18.


## How to Run



Install dependencies:



```bash

pip install -r requirements.txt

```

Run the training script:

```bash

python src/train.py

```

The script will:

\- download and prepare the Oxford Flowers dataset automatically if needed

\- build train/validation/test dataloaders

\- initialize the CNN model

\- train and validate the model

\- save the best model checkpoint to `checkpoints/best_flower_cnn.pth`

\- evaluate on the test set

\- display training curves and prediction examples

