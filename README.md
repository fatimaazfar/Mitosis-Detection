# Mitosis Classification | SOFTEC'24 AI Challenge Solution

This repository contains my solutions for the SOFTEC'24 AI Challenge, aimed at automating the classification of mitotic cells from provided image datasets. The competition was held in two rounds, each with its distinct set of challenges and requirements.

## Overview

Participants were tasked with developing models capable of accurately classifying cell images as mitotic or normal. This README details my approach to data preprocessing, model development and enhancement, and the results achieved in terms of validation F1 scores.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- PIL
- pandas

### Installation

To replicate this project, clone the repository.

## Dataset

The dataset provided by the competition organizers comprised images of cells, each labeled as either mitotic or normal. These images varied in dimensions and were supplied in JPEG format.

## Data Preprocessing

For both rounds, data preprocessing played a crucial role in preparing the images for the models. This included:

- Resizing images to a uniform dimension to ensure consistency in input size for the models.
- Applying normalization to standardize the pixel values across the dataset, improving model training efficiency.
- Data Augmentation: Rotation, Collor Jitter, Crop, Flip etc.
- Synthetic Data Generation to tackle with data imbalance: SMOTE

## Models

## Round 1

### Pre-trained VGG-11

- **Overview**: Adapted a pre-trained VGG-11 model for binary classification by customizing its final layer. This approach took advantage of the pre-trained model's robust feature extraction capabilities, enabling us to significantly cut down on training time and improve model accuracy.
- **Enhancements**: Incorporated Focal Loss to tackle the challenge posed by class imbalance, focusing the model's learning on more difficult, misclassified examples.
- **Training**: Used the Adam optimizer along with a learning rate scheduler to fine-tune the model effectively.

### Pre-trained DenseNet

- **Model**: Leveraged the DenseNet-121 model, known for its efficiency and compact architecture, as the foundation. The model was pre-trained on ImageNet, providing a robust starting point for feature extraction.
- **Enhancements**: Modified the classifier to suit binary classification, incorporating dropout for regularization. Implemented Focal Loss to address class imbalance, focusing the model's learning on harder-to-classify examples.
- **Training**: Employed Adam optimizer with learning rate scheduling for effective convergence.

### Custom CNN

- **Model**: Developed a SimpleCNN from scratch, designed specifically for the cell classification task. The architecture included convolutional layers followed by max-pooling, a flattening step, and fully connected layers, culminating in a binary output.
- **Enhancements**: Utilized class weights to compute the loss more effectively, addressing the imbalance between mitotic and normal cells within the dataset.
- **Training**: The model was trained using the BCEWithLogitsLoss, adjusted for class imbalance through weighting, and optimized using stochastic gradient descent (SGD).

## Round 2

### Pre-trained DenseNet with Enhancements

A pre-trained DenseNet model was adapted for the binary classification task, incorporating several enhancements to optimize performance:

- **Model Adaptation**: The DenseNet-121's classifier was modified to output a single value, corresponding to the binary classification task, with dropout added to prevent overfitting.
- **Focal Loss**: To address class imbalance, Focal Loss was implemented, prioritizing the learning on harder-to-classify examples and improving the robustness of the model.
- **Training Enhancements**:
  - **Optimizer**: Utilized Adam optimizer with a learning rate of 0.001, including weight decay for regularization.
  - **Scheduler**: A StepLR learning rate scheduler was employed to adjust the learning rate based on validation loss, aiding in overcoming plateaus during training.
  - **Early Stopping**: Introduced to halt training early if the validation loss did not improve after a specified number of epochs, preventing overfitting and saving computational resources.

### Pre-Trained Vision Transformers with Strategic Enhancements

Adopted a cutting-edge approach by leveraging pre-trained Vision Transformers (ViT) alongside a suite of strategic enhancements aimed at refining the model's performance on the given dataset. This approach was underpinned by advanced data augmentation techniques, the application of SMOTE for class balancing, and fine-tuning of the ViT model to suit our binary classification task. Below, we detail the methodologies and considerations that contributed to this comprehensive strategy.

#### **Advanced Data Augmentation**
To enhance the model's generalization capabilities and combat overfitting, we significantly expanded our data augmentation pipeline:

- **Comprehensive Transforms**: Included were standard operations such as resizing and random horizontal flips, supplemented with random rotations, random resized cropping, sharpness adjustments, and color jittering. These augmentations were designed to present the model with a broad spectrum of imaging conditions, thereby improving its robustness.
- **Normalization**: Following augmentation, images were normalized to align with the expected input distribution of the pre-trained ViT model, facilitating optimal learning dynamics.

#### Addressing Class Imbalance with SMOTE

Given the challenge posed by imbalanced classes in the dataset, we employed the Synthetic Minority Over-sampling Technique (SMOTE) to enhance minority class representation artificially, thereby promoting balanced training:

- **Application of SMOTE**: This technique was used to equalize the representation of classes in the training dataset, countering the potential bias towards the majority class.
- **Integration with DataLoader**: After applying SMOTE, the augmented data was converted back into tensors and introduced into a custom DataLoader, ensuring compatibility with our training regimen.

#### Model Selection and Adaptation: Vision Transformer (ViT)

The decision to utilize a ViT model was driven by its proven efficacy in image classification tasks, attributable to its self-attention mechanism:

- **Pre-trained Model Utilization**: We fine-tuned a pre-trained ViT (`vit_base_patch16_224`), adjusting its output layer to cater to our binary classification needs. This allowed us to leverage the transformer's extensive pre-learned features while tailoring it to our specific task.
- **Class Weighting in Loss Function**: To further mitigate the impact of class imbalance on model training, we computed class weights and employed them within a Focal Loss function, prioritizing the model's focus on more challenging classifications.

#### Training Process and Enhancements

Our training approach was characterized by precision and adaptability, incorporating several enhancements to optimize model training:

- **Optimization and Learning Rate Adjustment**: Utilizing the Adam optimizer with a carefully selected learning rate, complemented by a StepLR scheduler to dynamically adjust the learning rate based on performance metrics.
- **Model Performance Evaluation**: Throughout training, we meticulously monitored accuracy and F1 scores, ensuring that our enhancements effectively addressed the challenges presented by the dataset and task.

## License

This project is licensed under the MIT license.

Feel free to reach out to me on my [LinkedIn](linkedin.com/in/fatima-azfar-ziya-52a566154/) for any questions or collaboration.
