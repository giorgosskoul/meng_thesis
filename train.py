import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import Data
from utils.models import SquirtleNet, train, test
from utils.feature_extraction import extract_features, resize_transform
from utils.data_preparation import load_data, preprocess_data
import pandas as pd
import numpy as np
import os
import gc

# Set path and device
gc.collect()  # Forces garbage collection
DATASET_PATH = os.path.join(os.getcwd(), "ravdess_data")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU yields better performance

# Load dataset
print("Loading dataset...")
raw_audio, raw_labels = load_data(DATASET_PATH)
# Preprocess dataset
print("Preprocessing dataset...")
processed_audio, processed_labels = preprocess_data(raw_audio, raw_labels)
# Extract features
print("Extracting features...")
img_dim = 60
X = extract_features(processed_audio, fs=16000)
print(X.shape)
# Resize features
X_resized = []
for x in X:
    X_resized.append(resize_transform(x, img_dim, img_dim))
X = np.array(X_resized)

# Create dataset object
print(X.shape)
print("X shape:", X.shape)  # Should be (N, C, H, W)
print("Unique Labels:", np.unique(raw_labels))
# Add channel dimension (assuming grayscale input, 1 channel)
X = np.expand_dims(X, axis=1)
dataset = Data(X, pd.factorize(raw_labels)[0])

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
testloader = DataLoader(testset, batch_size=8, shuffle=False)

# Define model
model = SquirtleNet(num_channels=1, img_dim=img_dim,
                 conv1_output=128,conv1_kernel=2,conv1_stride=1,conv1_padding=1,
                 conv2_output=256,conv2_kernel=2,conv2_stride=1,conv2_padding=1,
                 conv3_output=390,conv3_kernel=2,conv3_stride=1,conv3_padding=1,     
                 pool1_kernel=2,pool1_stride=2,
                 pool2_kernel=2,pool2_stride=2,
                 fc1_output=2048,fc2_output=2048,
                 p_dropout1=0.5, p_dropout2=0.5,
                 print_option=1).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train model
print("Training model...")
epochs = 2
for epoch in range(epochs):
    train(model, device, trainloader, optimizer, criterion, epoch)

# Evaluate model
print("Evaluating model...")
test(model, device, testloader)
