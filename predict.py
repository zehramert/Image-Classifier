import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
import os
from util import load_checkpoint
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model on a flower dataset.")
    parser.add_argument('data_directory', type=str, help='Directory for the dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'vgg13', 'resnet50'],
                        help='Choose model architecture')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes (default: 1)')
    parser.add_argument('--category_names', type=str, default=None,
                        help='Path to the JSON file mapping category labels to flower names')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('input', type=str, help="Path to the image file for prediction")
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')

    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    img = Image.open(image)

    img = img.resize((256, 256))

    width, height = img.size

    # Crop out the center
    left = (width - 224) / 2
    right = (width + 224) / 2
    top = (height - 224) / 2
    bottom = (height + 224) / 2

    img = img.crop((left, top, right, bottom))  # order matters!

    np_image = np.array(img) / 255.0  # Convert from [0, 255] to [0, 1]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    np_image = np_image.transpose((2, 0, 1))  # Reorder to dimensions

    # Convert NumPy array to PyTorch tensor
    tensor_image = torch.from_numpy(np_image).float()  # Convert to float tensor

    return tensor_image


def predict(image_path, model, device, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file

    image = process_image(image_path)

    image = image.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        output = model(image)

    probability = F.softmax(output, dim=1)

    # Get top k probabilities and indices
    probs, indices = probability.topk(topk)

    # Convert numpy arrays
    probs = probs.cpu().numpy().flatten()  # flattens the array into a 1D array
    indices = indices.cpu().numpy().flatten()

    # Map indices to class labels
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [index_to_class[idx] for idx in indices]

    if cat_to_name:
        top_classes = [cat_to_name.get(cls, cls) for cls in top_classes]





    return probs, top_classes


def main():
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        cat_to_name = None

    model, optimizer = load_checkpoint(args.checkpoint)
    model = model.to(device)

    probs, class_names = predict(args.input, model, device, cat_to_name, args.top_k)

    for name, prob in zip(class_names, probs):
        print(f'Prediction: {name} with probability: {prob*100:.2f}%')

if __name__ == "__main__":
    main()
