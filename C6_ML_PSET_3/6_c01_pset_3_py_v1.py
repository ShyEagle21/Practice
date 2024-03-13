import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

torch.manual_seed(0)

class ChessDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = []
        self.labels = []
        
        for label, subdir in enumerate(['insufficient_material', 'sufficient_material']):
            subdir_path = os.path.join(directory, subdir)
            for filename in os.listdir(subdir_path):
                if filename.endswith('.png'):
                    self.images.append(os.path.join(subdir_path, filename))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# load a pretrained ResNet model
model = models.resnet18(weights='ResNet18_Weights.DEFAULT')

# create the dataset
train_dataset = ChessDataset('chess_dataset/train', transform=data_transforms['train'])
val_dataset = ChessDataset('chess_dataset/val', transform=data_transforms['val'])

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


###########################
##  Beginning of Part a  ##
###########################


# TODO: modify the fully connected layer for binary classification
# hint: it might be helpful to print(model) to see how the model is structured

# use a GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


###########################
##  Beginning of Part b  ##
###########################


def train(epoch_num):
    # set the model to train mode
    model.train()

    running_loss = 0.0
    total_loss = 0.0
    running_count = 0
    total_count = 0      

    for batch_index, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        '''
        TODO: complete the training loop. You will need to do the following:
        1. zero the parameter gradients
        2. run a forward pass
        3. run a backwards pass and optimizer step
        '''

        # update loss and count
        running_loss += loss.item() * inputs.size(0)
        total_loss += loss.item() * inputs.size(0)

        running_count += inputs.size(0)
        total_count += inputs.size(0)

        # print every 50 mini-batches
        if batch_index % 50 == 49:
            print('[%d, %5d] avg batch loss: %.3f avg epoch loss: %.3f' %
                (epoch_num + 1, batch_index + 1, running_loss / running_count, total_loss / total_count))
            running_loss = 0.0
            running_count = 0


def validate():
    # set the model to evaluation mode
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    # no need to track gradients for validation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # TODO: run the forward pass (same as part c)

            # TODO: from your output (which are probabilities for each class, find the predicted
            # class)
            correct_count = None

            # update loss and count
            total_loss += loss.item() * labels.size(0)
            total_correct += correct_count
            total_count += labels.size(0)

    accuracy = 100 * total_correct / total_count
    print()
    print(f"Evaluation loss: {total_loss / total_count :.3f}")
    print(f'Accuracy of the model on the validation images: {accuracy: .2f}%')
    print()


###########################
##  Beginning of Part c  ##
###########################


# TODO: validate your model before training

num_epochs = 1
# TODO: train and validate your model