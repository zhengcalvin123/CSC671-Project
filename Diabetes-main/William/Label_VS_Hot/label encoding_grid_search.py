import torch
import torch.nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from skorch import NeuralNetClassifier
import matplotlib.pyplot as plt
import timeit

# Load the dataset
df_data = pd.read_csv("diabetes_data_upload.csv")
test_data = df_data

# Extract the feature 
feature_names = test_data.columns[1:]

# Initialize LabelEncoder
label_encoder = LabelEncoder()
# Fit and transform the 'Gender' column
for feature in feature_names:
    test_data[feature] = label_encoder.fit_transform(test_data[feature])

train_x = test_data.iloc[:,0:-1].values # convert to numpy 1d array 

# Extract the target 
y_panda = test_data['class']
y_numpy = y_panda.to_numpy()
train_y = y_numpy.reshape(-1,1)

data_features=train_x.shape[1] #16
data_labels=train_y.shape[1] #1

#***************************************************#

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
# print(device)

model = NeuralNetClassifier(
    module=MLP,
    criterion=torch.nn.BCEWithLogitsLoss(),
    device=device,
    module__num_features=data_features,
    module__num_classes=data_labels,
    iterator_train__shuffle=True,
    callbacks=[Checkpoint()],
    verbose=0
)

epochs = [100,200,300]
batch_size = [10,20,30]
learning_rate = [0.1,0.01]
num_layers = range(1)

param_grid = {
    "max_epochs": epochs,
    "batch_size": batch_size,
    "lr": learning_rate,
    "optimizer": [torch.optim.SGD, torch.optim.Adam, torch.optim.Adadelta],
    "module__num_layers": num_layers
}

# print(model.initialize())

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    # verbose=4,
    # error_score='raise'
)


start = timeit.default_timer()
grid_search.fit(train_x.astype(np.float32), train_y.astype(np.float32))
stop = timeit.default_timer()

elapsed_time = stop - start
print("Elapsed Time:", elapsed_time)

print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_score_)

df = pd.DataFrame(grid_search.cv_results_)
df = df.sort_values("rank_test_score")
df.to_csv("csv_result.csv")