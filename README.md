\# Oxford 102 Flowers Classification with CNN



This project implements a convolutional neural network (CNN) in PyTorch for image classification on the Oxford 102 Flowers dataset. The project includes data preprocessing, model training, validation, test evaluation, and visualization of training results and sample predictions.



\## Project Goal



The goal of this project is to build a custom CNN baseline for multi-class flower image classification and analyze its performance through training curves and prediction examples.



\## Dataset



This project uses the Oxford 102 Flowers dataset.



\- Number of classes: 102

\- Task: multi-class image classification

\- Data split:

&#x20; - Training set: 70%

&#x20; - Validation set: 15%

&#x20; - Test set: 15%



The dataset is automatically downloaded in the script.



\## Model and Method



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



\## Training Configuration



\- Framework: PyTorch

\- Optimizer: Adam

\- Loss function: CrossEntropyLoss

\- Epochs: 10

\- Batch size: 32

\- Learning rate: 5e-4

\- Weight decay: 5e-4

\- Dropout: 0.5

\- Random seed: 42



\## Results



The model was successfully trained on the Oxford 102 Flowers dataset and showed stable improvement during training.



Final performance of the best saved model:



\- Best validation accuracy: 53.91%

\- Test loss: 1.7108

\- Test accuracy: 54.92%



\### Loss Curve

!\[Loss Curve](outputs/figures/loss\_curve.png)



\### Accuracy Curve

!\[Accuracy Curve](outputs/figures/accuracy\_curve.png)



\### Sample Predictions

!\[Prediction Examples](outputs/figures/prediction\_examples.png)



\## How to Run



Install dependencies:



```bash

pip install -r requirements.txt

```

Run the training script:

```bash

python src/train.py

```

The script will:

\- download the dataset automatically if needed

\- train the CNN model

\- save the best model checkpoint

\- evaluate on the test set

\- display training curves and prediction examples

