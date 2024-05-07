import torch
import torch.nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Load the dataset
df_data = pd.read_csv("diabetes_data_upload.csv")

# Extract the feature and target
feature_names = df_data.columns[1:]

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the features
for feature in feature_names:
    df_data[feature] = label_encoder.fit_transform(df_data[feature])

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(df_data.iloc[:, 0:-1])
y = df_data['class'] # Encoding binary labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Convert to PyTorch tensors
x_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for the model
xtest_tensor = torch.tensor(X_test, dtype=torch.float32)
ytest_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for the model

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
model = MLPModel(num_features=X_train.shape[1])

# Define loss and optimizer
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
train_loss = []
test_loss = []
total_accuracy = 0.0  # Initialize total accuracy
for epoch in range(num_epochs):
    epoch_train_loss = 0.0
    epoch_test_loss = 0.0
    num_batches_train = 0
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        num_batches_train += 1
    
    # Calculate average training loss per epoch
    epoch_train_loss /= num_batches_train
    train_loss.append(epoch_train_loss)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(xtest_tensor)
        loss = criterion(test_outputs, ytest_tensor)
        epoch_test_loss = loss.item()
        test_loss.append(epoch_test_loss)

    # Calculate test accuracy
    test_outputs = (test_outputs >= 0.5).float()  # Convert probabilities to binary predictions
    accuracy = (test_outputs == ytest_tensor).float().mean().item()
    total_accuracy += accuracy  # Accumulate accuracy for each epoch
    
    # Print and display test accuracy and loss
    print(f"Epoch {epoch+1}/{num_epochs},Train Loss: {epoch_train_loss},Test Loss: {epoch_test_loss},Test Accuracy: {accuracy}")

# Calculate average accuracy
average_accuracy = total_accuracy / num_epochs
print(f"Average Accuracy: {average_accuracy}")

# Plot both training and test loss
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(ytest_tensor.numpy(), (test_outputs >= 0.5).numpy())
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