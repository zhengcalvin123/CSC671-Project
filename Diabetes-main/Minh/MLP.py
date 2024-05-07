import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score


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


def compute_accuracy(model, dataloader):
    model = model.eval()

    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.inference_mode():  # same as torch.no_grad
            logits = model(features)

        predictions = (logits > 0.5).float()
        compare = torch.eq(predictions, labels).float()
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples


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
    # print(f"Train Acc {train_acc * 100:.2f}% | Val Acc {val_acc * 100:.2f}%")

epochs = list(range(0, len(losses)))

train_acc = compute_accuracy(model, train_loader)
val_acc = compute_accuracy(model, val_loader)
test_acc = compute_accuracy(model, test_loader)

print(f"Train Acc {train_acc*100:.2f}%")
print(f"Val Acc {val_acc*100:.2f}%")
print(f"Test Acc {test_acc*100:.2f}%")

plt.title("Training Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.plot(epochs, losses)
plt.show()
