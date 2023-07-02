import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Defining the CNN model
class MineralClassifier(nn.Module):
    def __init__(self):
        super(MineralClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 50 * 50, 128)
        self.fc2 = nn.Linear(128, 7)  # Assuming 7 mineral classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 50 * 50)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Defining the custom dataset class
class MineralDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# Loading and preprocessng the dataset
dataset_dir = 'C:\\Users\\Oiiis\\Desktop\\mining-py\\minet'  # Parent directory containing the mineral directories
mineral_labels = os.listdir(dataset_dir)  # List of mineral labels

images = []
labels = []

for label_idx, mineral_label in enumerate(mineral_labels):
    label_dir = os.path.join(dataset_dir, mineral_label)
    if os.path.isdir(label_dir):
        image_files = os.listdir(label_dir)
        for image_file in image_files:
            image_path = os.path.join(label_dir, image_file)
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (100, 100))  # Resize the image to a fixed size
                    images.append(image)
                    labels.append(label_idx)
                else:
                    print(f"Failed to load image: {image_path}")
            except Exception as e:
                print(f"Error loading image: {image_path}")
                print(str(e))
    else:
        print(f"Invalid directory: {label_dir}")

# Converting the image and label lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Converting the data to PyTorch tensors
X_train = torch.tensor(X_train.transpose((0, 3, 1, 2)), dtype=torch.float32)
X_test = torch.tensor(X_test.transpose((0, 3, 1, 2)), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Creating data loaders for training and testing sets
train_dataset = MineralDataset(X_train, y_train)
test_dataset = MineralDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Creating the model and optimizer
model = MineralClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluating the model
model.eval()
test_predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_predictions.extend(predicted.tolist())

test_predictions = np.array(test_predictions)
accuracy = accuracy_score(y_test.numpy(), test_predictions)
print("Accuracy:", accuracy)
print(classification_report(y_test.numpy(), test_predictions, target_names=mineral_labels))
