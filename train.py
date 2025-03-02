import argparse
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from util import save_checkpoint
from torch.utils.data import DataLoader


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
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    return parser.parse_args()

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),  # To resize and crop
        transforms.RandomHorizontalFlip(),  # To flip randomly wiht the probability of 0.5
        transforms.RandomRotation(30)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])

    test_validation_transforms = transforms.Compose([
        transforms.Resize(226),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train_dataset': datasets.ImageFolder(root=train_dir, transform=train_transforms),
        'test_dataset': datasets.ImageFolder(root=test_dir, transform=test_validation_transforms),
        'validation_dataset': datasets.ImageFolder(root=valid_dir, transform=test_validation_transforms)
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train_dataset'], batch_size=32, shuffle=True, num_workers=4),
        'test': DataLoader(image_datasets['test_dataset'], batch_size=32, shuffle=False, num_workers=4),
        'val': DataLoader(image_datasets['validation_dataset'], batch_size=32, shuffle=False, num_workers=4)
    }

    return image_datasets, dataloaders


def build_model(arch='vgg16', hidden_units=512):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError("Invalid model architecture. Choose from 'vgg16', 'vgg13', or 'resnet50'.")

    for param in model.parameters():
        param.requires_grad = False  # Freeze model parameters

    # Modify classifier
    if arch == 'resnet50':
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102)
        )
    else:
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )

    return model


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=5, device='gpu'):

    print("Starting training...")

    print("Checking dataset sizes...")
    print(f"Train dataset size: {len(dataloaders['train'].dataset)}")
    print(f"Validation dataset size: {len(dataloaders['val'].dataset)}")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}...")

        running_loss = 0
        val_loss = 0
        correct = 0
        total = 0

        # Training Phase
        model.train()
        for inputs, labels in dataloaders['train']:
            # print("Training Batch Loaded")  # debug
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # print(f"Loss before backward: {loss.item()}")  # Debug loss

            try:
                loss.backward()  # Check if this is the issue
                optimizer.step()
            except RuntimeError as e:
                print(f"Error in backward pass: {e}")  # Debugging

            # Debug optimizer step
            #print("Updating model weights...")
            #for param_group in optimizer.param_groups:
                #print(f"Learning rate: {param_group['lr']}")

            running_loss += loss.item()

        # Validation Phase
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)


        scheduler.step()

        # Compute Loss & Accuracy
        train_loss = running_loss / len(dataloaders['train'])
        val_loss /= len(dataloaders['val'])
        val_accuracy = correct / total

        print(f"Train Loss: {train_loss:.4f}.. "
              f"Val Loss: {val_loss:.4f}.. "
              f"Val Accuracy: {val_accuracy:.4f}", flush=True)

    print("Training complete!")
    return model


def main():
    args = parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    image_dataset, dataloaders = load_data(args.data_directory)

    trainloader = dataloaders['train']
    validloader = dataloaders['val']

    model = build_model(arch=args.arch, hidden_units=args.hidden_units)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    model = train_model(model, loss_function, optimizer, scheduler, dataloaders, args.epochs)

    # Save the model checkpoint
    save_checkpoint(model, optimizer, model.classifier, image_dataset, args.epochs, args.learning_rate, args.save_dir)


if __name__ == "__main__":
    main()



