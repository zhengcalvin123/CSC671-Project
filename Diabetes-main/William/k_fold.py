import torch
import torch.nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df_data = pd.read_csv("diabetes_data_upload.csv")

# Extract the feature and target
feature_names = df_data.columns[1:]

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the features
for feature in feature_names:
    df_data[feature] = label_encoder.fit_transform(df_data[feature])

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(df_data.iloc[:, 1:-1])
y = df_data['class'].replace({'Positive': 1, 'Negative': 0})  # Encoding binary labels

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

# Define loss and optimizer
criterion = torch.nn.BCELoss()

# Training loop
num_epochs = 100

num_folds = kf.get_n_splits(X)
num_rows = num_folds // 2 + num_folds % 2
fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, num_rows * 5))

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    model = MLPModel(num_features=X.shape[1])  # Initialize the model for each fold
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define optimizer for each fold
    
    train_loss = []  # Store train loss for each fold
    test_losses = []  # Store test loss for each fold
    test_accuracy = []  # Store test accuracy for each fold
    
    x_train_fold, x_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    
    x_train_tensor = torch.tensor(x_train_fold, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_fold.values, dtype=torch.float32).unsqueeze(1)
    x_test_tensor = torch.tensor(x_test_fold, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_fold.values, dtype=torch.float32).unsqueeze(1)
    
    train_ds = MyDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

    # Evaluation
    test_ds = MyDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_ds, batch_size=32, shuffle=False)  # No need to shuffle test data

    test_outputs = []  # Store test predictions

    with torch.no_grad():
        model.eval()
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_losses.append(loss.item())
            predictions = (outputs >= 0.5).float()
            accuracy = (predictions == targets).float().mean().item()
            test_accuracy.append(accuracy)
            test_outputs.extend(predictions.squeeze().cpu().numpy())

    # Calculate overall test loss and accuracy
    overall_test_loss = np.mean(test_losses)
    overall_test_accuracy = np.mean(test_accuracy)

    print(f"Fold {fold+1}, Overall Test Loss: {overall_test_loss}, Overall Test Accuracy: {overall_test_accuracy}")

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test_tensor.numpy(), np.array(test_outputs))
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot confusion matrix with annotations
    row = fold // 2
    col = fold % 2
    ax = axs[row, col]
    ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix - Fold {fold+1}')
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

# If there's an empty subplot, remove it
if num_folds % 2 != 0:
    fig.delaxes(axs[num_rows - 1, 1])

# Plot the figure
plt.tight_layout()
plt.show()


# Plot both training and test loss
plt.plot(train_loss, label='Train Loss')
test_loss_iteration = np.linspace(0, num_epochs * len(train_loss) // num_epochs, len(test_losses), endpoint=False)
plt.plot(test_loss_iteration, test_losses, marker='o', linestyle='', label='Test Loss')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.show()