import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd

"""
this code doesn't work because this dataset isn't sequential so it doesn't apply to the RNN structure 
and it is too late to figure that out
"""

torch.manual_seed(123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# data preprocessing
data_features = pd.read_csv("diabetes_data_upload.csv")
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
    x_values, y_values, test_size=0.2, random_state=1, stratify=y_values)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=1, stratify=y_train)


class RNN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layer, bidirectional, dropout=0):
        super().__init__()

        if num_layer <= 1:
            dropout = 0
            num_layer = 1

        self.hidden_size = hidden_size
        self.num_layer = num_layer

        # process the vector sequences
        self.lstm = torch.nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layer,
                                  bidirectional=bidirectional,
                                  batch_first=True,
                                  dropout=dropout)

        # fully connected layer to predict
        self.fc = torch.nn.Linear(hidden_size, output_size)

        # force all prediction to be between 0-1
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        print(x.size())

        h0 = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device)
        print(h0.shape, c0.shape)
        output, _ = self.lstm(x, (h0, c0))
        output = self.fc(output[:, -1, :])
        output = self.sigmoid(output)

        return output


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

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print("batch test", x_batch.shape, y_batch.shape)
    break


model = RNN(input_size=x_values.shape[1], output_size=1, hidden_size=128, num_layer=1, bidirectional=1)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

losses = []
num_epochs = 200


def train_one_epoch():
    model.train(True)
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        features, labels = batch[0].to(device), batch[1].to(device)
        logits = model(features)

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate_one_epoch():
    model.train(False)
    total_loss = 0.0

    for batch_idx, batch in enumerate(val_loader):
        features, labels = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            logits = model(features)
            loss = F.binary_cross_entropy_with_logits(logits, features)
            total_loss += loss

    print(f"average loss: {total_loss / len(val_loader):.2f}")


for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()
