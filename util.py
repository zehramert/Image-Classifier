import json
import torch
import torch.optim as optim
from torchvision import models


def save_checkpoint(trained_model, optimizer, classifier, image_datasets, epochs, learning_rate, checkpoint_path='checkpoint.pth'):
    trained_model.class_to_idx = image_datasets['train_dataset'].class_to_idx

    checkpoint = {
        'epochs': epochs,
        'model_state_dict': trained_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': trained_model.class_to_idx,
        'classifier': classifier,
        'arch': 'vgg16',
        'learning_rate': learning_rate,
    }

    torch.save(checkpoint, checkpoint_path)  # Save the checkpoint
    print("Model saved successfully!")


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = getattr(models, checkpoint['arch'])(pretrained=True)

    learning_rate = checkpoint['learning_rate']
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'], weight_decay=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Optionally return optimizer along with model if continuing training
    return model, optimizer



