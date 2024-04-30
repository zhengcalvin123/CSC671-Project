import torch
import torch.nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df_data = pd.read_csv("diabetes_data_upload.csv")

test_data = df_data

# Extract the feature and target
feature_names = test_data.columns[1:]

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the features
for feature in feature_names:
    test_data[feature] = label_encoder.fit_transform(test_data[feature])

# Train & Test Set
X = test_data.iloc[:, 1:-1]
y = test_data['class'].replace({'Positive': 1, 'Negative': 0})  # Encoding binary labels

train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=42)

# Standardize the data
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Convert to PyTorch tensors
x_tensor = torch.tensor(train_x, dtype=torch.float32)
y_tensor = torch.tensor(train_y.values, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for the model
xtest_tensor = torch.tensor(test_x, dtype=torch.float32)
ytest_tensor = torch.tensor(test_y.values, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for the model

# Define the MLP model
class MLPModel(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.all_layers = torch.nn.Sequential(
            torch.nn.Linear(num_features, 30),
            torch.nn.Sigmoid(),
            torch.nn.Linear(30, 20),
            torch.nn.Sigmoid(),
            torch.nn.Linear(20, 1),  # Output dimension changed to 1 for binary classification
            torch.nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        logits = self.all_layers(x)
        return logits

# Define the dataset and DataLoader
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.labels)

# Initialize dataset and DataLoader
train_ds = MyDataset(x_tensor, y_tensor)
train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)

# Initialize the model
model = MLPModel(num_features=train_x.shape[1])

# Define loss and optimizer
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
train_loss=[]
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

plt.plot(train_loss)
plt.show()


# Evaluation
with torch.no_grad():
    model.eval()
    test_outputs = model(xtest_tensor)
    test_loss = criterion(test_outputs, ytest_tensor)
    test_outputs = (test_outputs >= 0.5).float()  # Convert probabilities to binary predictions
    accuracy = (test_outputs == ytest_tensor).float().mean().item()
    print(f"Test Loss: {test_loss.item()}, Test Accuracy: {accuracy}")

    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(ytest_tensor.numpy(), test_outputs.numpy())
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Plot confusion matrix with annotations
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

