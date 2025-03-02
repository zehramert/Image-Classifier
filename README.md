# Image Classifier

This project implements a deep learning-based flower classification system using PyTorch. The model is trained on a dataset of flower images and uses a convolutional neural network (CNN) architecture to predict the class of a given flower image.

## Project Overview

The goal of this project is to classify images of flowers into different species using a pre-trained deep learning model, fine-tuned for flower classification. The system supports:

_Model Selection_: Choose between different model architectures such as VGG16, VGG13, or ResNet50.

_Top-K Predictions_: Return the top K most probable flower species.

_GPU Support_: Utilize GPU for faster training and inference if available.

_Category Mapping_: Map numerical class indices to human-readable flower names.

## Features

_Image Preprocessing_: Scales, crops, and normalizes input images for model prediction.

_Prediction_: Predicts the top K flower species for a given image.

_Model Training & Checkpointing_: Train a model on a flower dataset and save the trained model for later use.

_Prediction from a Checkpoint_: Load a pre-trained model and perform inference on a new image.


## Model Architecture

This project provides the following architectures for training:

- VGG16
- VGG13
- ResNet50
  
Each model architecture is pre-trained on ImageNet and can be fine-tuned on the flower dataset.


## Usage
__Training the Model:__
```
python flower_classifier.py <data_directory> --arch <model_architecture> --epochs <num_epochs> --learning_rate <learning_rate> --save_dir <save_checkpoint_path>

```

__Predicting with a Pre-Trained Model:__

```
python flower_classifier.py --input <image_path> --checkpoint <checkpoint_path> --category_names <category_names_json> --top_k <top_k> --gpu

```

__Example Command:__
```
python flower_classifier.py --input 'flowers/test_image.jpg' --checkpoint 'checkpoint.pth' --category_names 'cat_to_name.json' --top_k 3 --gpu
```
