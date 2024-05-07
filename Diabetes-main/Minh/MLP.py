import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchmetrics
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torchmetrics.classification import BinaryPrecision

torch.manual_seed(123)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# data preprocessing
data_features = pd.read_csv("..\\diabetes_data_upload.csv")
data_labels = data_features.loc[:, ["class"]]
data_features = data_features.drop("class", axis=1)

# encoding categorical value with numerical or boolean value
data_features_encoded = pd.get_dummies(data_features, drop_first=True, dtype=float)
data_labels_encoded = pd.get_dummies(data_labels, drop_first=True, dtype=float)

x_values = data_features_encoded.values
y_values = data_labels_encoded.values

x_mean = x_values.mean(axis=0)
x_std = x_values.std(axis=0)

# z-scale transformation
x_normal = (x_values - x_mean) / x_std

x_train, x_test, y_train, y_test = train_test_split(
    x_normal, y_values, test_size=0.2, random_state=1, stratify=y_values)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=1, stratify=y_train)


class MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        # using non-linear relu activation function for faster convergence

        self.all_layers = torch.nn.Sequential(
            # using non-linear activation sigmoid

            # 1st hidden layer
            torch.nn.Linear(num_features, 25),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(25),
            torch.nn.Dropout(p=0.5),

            # 2nd hidden layer
            torch.nn.Linear(25, 25),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(25),
            torch.nn.Dropout(p=0.5),

            # 3rd hidden layer
            torch.nn.Linear(25, num_classes)
        )

    def forward(self, x):
        logits = self.all_layers(x)
        return logits


class EarlyStopping(object):
    def __init__(self, patience=5, min_delta=0.001):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta  # minimum change requirement
        self.min_validation_loss = float('inf')
        self.counter = 0

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class MyDataSet(Dataset):
    def __init__(self, x, y):
        self.features = torch.tensor(x, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return self.labels.shape[0]


train_ds = MyDataSet(x_train, y_train)
val_ds = MyDataSet(x_val, y_val)
test_ds = MyDataSet(x_test, y_test)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=10,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_ds,
    batch_size=10,
    shuffle=False,
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=10,
    shuffle=False,
)


def compute_loss(model, dataloader):
    model = model.eval()

    total_loss = 0.0
    total_example = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.inference_mode():
            logits = model(features)

        # calculate mean squared error (MSE) between predictions and labels
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        total_loss += torch.mean(loss).item() * len(labels)

        total_example += len(labels)

    # return average loss as accuracy metric for regression
    return total_loss / total_example


def compute_metrics(model, dataloader):
    model = model.eval()

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.inference_mode():
            # same as torch.no_grad
            logits = model(features)

        predictions = (logits > 0.5).float()

        true_positives += torch.sum((predictions == 1) & (labels == 1))
        false_positives += torch.sum((predictions == 1) & (labels == 0))
        true_negatives += torch.sum((predictions == 0) & (labels == 0))
        false_negatives += torch.sum((predictions == 0) & (labels == 1))

    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + true_negatives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score


model = MLP(num_features=data_features.shape[1], num_classes=1)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
early_stopping = EarlyStopping(patience=25, min_delta=0.0001)

num_epochs = 200

losses = []

for epoch in range(num_epochs):

    model = model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # update model parameters
        optimizer.step()

    train_loss = compute_loss(model, train_loader)
    losses.append(train_loss)

    val_loss = compute_loss(model, val_loader)
    if early_stopping.early_stop(val_loss):
        break

epochs = list(range(0, len(losses)))

train_metrics = compute_metrics(model, train_loader)
val_metrics = compute_metrics(model, val_loader)
test_metrics = compute_metrics(model, test_loader)

print(f"Train Accuracy {train_metrics[0] * 100:.2f}% \n"
      f"Train Precision {train_metrics[1] * 100:.2f}% \n"
      f"Train Recall {train_metrics[2] * 100:.2f}% \n"
      f"Train f1 score {train_metrics[3] * 100:.2f}%")

print(f"Validation Accuracy {val_metrics[0] * 100:.2f}% \n"
      f"Validation Precision {val_metrics[1] * 100:.2f}% \n"
      f"Validation Recall {val_metrics[2] * 100:.2f}% \n"
      f"Validation f1 score {val_metrics[3] * 100:.2f}%")

print(f"Test Accuracy {test_metrics[0] * 100:.2f}% \n"
      f"Test Precision {test_metrics[1] * 100:.2f}% \n"
      f"Test Recall {test_metrics[2] * 100:.2f}% \n"
      f"Test f1 score {test_metrics[3] * 100:.2f}%")

plt.title("Training Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.plot(epochs, losses)
plt.show()
