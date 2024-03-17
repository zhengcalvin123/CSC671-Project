import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, Checkpoint

# data preprocessing
data_features = pd.read_csv("..\\diabetes_data_upload.csv")
data_labels = data_features.loc[:, ["class"]]
data_features = data_features.drop("class", axis=1)

# encoding categorical value with numerical or boolean value
data_features_encoded = pd.get_dummies(data_features, drop_first=True, dtype=int)
data_labels_encoded = pd.get_dummies(data_labels, drop_first=True, dtype=int)

# z-score standardization scaler x = (x - u) / s
scaler = StandardScaler()

x_train = data_features_encoded
y_train = data_labels_encoded.to_numpy()

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_train = scaler.fit_transform(x_train)


class MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers):
        super().__init__()
        # using non-linear sigmoid activation function

        # minimum 3 layers in the system
        num_layers = max(num_layers, 1)

        # 1st and last layer are always the same
        modules = []
        modules.append(torch.nn.Linear(num_features, 50))
        modules.append(torch.nn.Sigmoid())
        for i in range(num_layers):
            modules.append(torch.nn.Linear(50, 50))
            modules.append(torch.nn.Sigmoid())
        modules.append(torch.nn.Linear(50, num_classes))

        self.all_layers = torch.nn.Sequential(*modules)

    def forward(self, x):
        logits = self.all_layers(x)
        return logits


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# applying all best parameters from GridSearchCV hyperparameters tuning
model = NeuralNetClassifier(
    module=MLP,
    criterion=torch.nn.BCEWithLogitsLoss(),  # using this over BCELoss due to no activation function for last layer
    batch_size=100,
    lr=0.1,
    max_epochs=200,
    device=device,
    optimizer=torch.optim.Adam,
    module__num_features=data_features.shape[1],
    module__num_classes=data_labels.shape[1],
    module__num_layers=1,
    callbacks=[
        EpochScoring(scoring='accuracy', name='train_acc', on_train=True),
        Checkpoint()
    ],
    verbose=0
)

model.fit(x_train, y_train)


# converting values into boolean values
def to_binary_labels(y):
    return torch.tensor(y > 0.5).float()


# print(model.history)

# epochs = model.history[:, "epoch"]

train_loss = model.history[:, "train_loss"]
valid_loss = model.history[:, "valid_loss"]

train_acc = model.history[:, "train_acc"]
valid_acc = model.history[:, "valid_acc"]
print(f"Training accuracy: {train_acc[-1]*100:.2f}%")
print(f"Validation accuracy: {valid_acc[-1]*100:.2f}%")

plt.plot(train_loss, 'o-', label='training')
plt.plot(valid_loss, 'o-', label='validation')
plt.legend()
plt.show()

plt.plot(train_acc, 'o-', label='training')
plt.plot(valid_acc, 'o-', label='validation')
plt.legend()
plt.show()
