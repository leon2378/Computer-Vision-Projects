import torch
import pandas as pd
import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet152, ResNet152_Weights
#from torchvision.transforms import functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report#, roc_curve
import cv2

#class ConvertToRGB:
#    def __call__(self, image):
#        return F.to_grayscale(image, num_output_channels=3) if image.mode != 'RGB' else image

class SportsClassificationDataset(Dataset):
    def __init__(self, root_dir, csv_file, dataset_type, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.transform = transform

        self.annotations = self.annotations[self.annotations['data set'] == self.dataset_type]

        self.label_to_idx = {label: idx for idx, label in enumerate(self.annotations['labels'].unique())}
        print(f'Number of unique Clases: {self.label_to_idx} for {self.dataset_type} set')

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['filepaths'])
        
        try:
            # Use OpenCV to read the image
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError
            
            # Convert the image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the image to PIL Image for consistency with transforms
            image = Image.fromarray(image)

            #image = Image.open(img_path)

            if self.transform:
                image = self.transform(image)

            label = self.annotations.iloc[idx]['labels']
            label_to_idx = self.label_to_idx[label] # convert labels to integers

            return image, label_to_idx

        except (FileNotFoundError, UnidentifiedImageError):
            print(f'File not found: {img_path}. Skipping this file')

            return self.__getitem__((idx +1) % len(self))
        
        #label = self.annotations.iloc[idx]['labels']
        #label_to_idx = self.label_to_idx[label] # convert labels to integers


        #if self.transform:
        #    image = self.transform(image)

        #return image, label_to_idx
    
transform = transforms.Compose([
    #ConvertToRGB(),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=0.1),
    transforms.ToTensor()])

# define file paths
root_dir = "C:/Users/Leon-PC/Downloads/Projects/computer vision/Image Classification/100 Sports Image Classification"

csv_path = "sports.csv"

# define the datasets
train_dataset = SportsClassificationDataset(csv_file=csv_path, root_dir=root_dir, dataset_type='train', transform=transform)
valid_dataset = SportsClassificationDataset(csv_file=csv_path, root_dir=root_dir, dataset_type='valid', transform=transform)
test_dataset = SportsClassificationDataset(csv_file=csv_path, root_dir=root_dir, dataset_type='test', transform=transform)   


# define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
print("Datasets Found")

# verify the datasets sizes
print(f'Size of train dataset: {len(train_dataset)}')
print(f'Size of valid dataset: {len(valid_dataset)}')
print(f'Size of test dataset: {len(test_dataset)}')


num_classes = len(train_dataset.label_to_idx)
print(f'Number of classes: {num_classes}')

model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)#, ignore_mismatched_sizes=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using Device: {device}')

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)    

num_epochs = 250

for epoch in range(num_epochs):
    print(f'Epoch: {epoch+1}/{num_epochs}')
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)

        # forward pass
        outputs = model(images)#.logits
        loss = criterion(outputs, labels)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch: [{epoch+1/num_epochs}], Training loss: {running_loss}')

    # validation loop
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)#.logits
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_loss /= len(valid_loader)
    valid_accuracy = 100*correct/total

    print(f'Validation loss: {valid_loss}, Validation Accuracy: {valid_accuracy}%')

print("Training Completed")

# test loop
model.eval()
test_loss = 0
correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing', leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)#.logits
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss /= len(test_loader)
test_accuracy = 100*correct/total

print(f'Testing Loss: {test_loss}, Testing Accuracy: {test_accuracy}%')

# generate classification report 
labels = list(range(num_classes)) # ensuring labels match the number of classes

class_report = classification_report(all_labels, all_preds, labels=labels, target_names=list(train_dataset.label_to_idx.keys()), zero_division=1)
print("\n Classification Report: \n", class_report)

# save classification report to a flie
report_path = 'classification_report.txt'

with open(report_path, 'w') as f:
    f.write(class_report) # type: ignore

print(f'\n Classification Report saved to: {report_path}')
