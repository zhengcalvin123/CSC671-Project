import timeit
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint

# data preprocessing
data_features = pd.read_csv("..\\diabetes_data_upload.csv")
data_labels = data_features.loc[:, ["class"]]
data_features = data_features.drop("class", axis=1)

# encoding categorical value with numerical or boolean value
data_features_encoded = pd.get_dummies(data_features, drop_first=True, dtype=int)
data_labels_encoded = pd.get_dummies(data_labels, drop_first=True, dtype=int)

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

model = NeuralNetClassifier(
    module=MLP,
    criterion=torch.nn.BCEWithLogitsLoss(),
    device=device,
    module__num_features=data_features.shape[1],
    module__num_classes=data_labels.shape[1],
    iterator_train__shuffle=True,
    callbacks=[Checkpoint()],
    verbose=0
)

epochs = [100, 200, 300, 400, 500]
batch_size = [20, 40, 60, 80, 100]
learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001]
num_layers = range(1, 5)

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

# print(x_train.shape)
# print(y_train.shape)

start = timeit.default_timer()
grid_search.fit(x_train.astype(np.float32), y_train.astype(np.float32))
print(timeit.default_timer() - start)

print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_score_)

df = pd.DataFrame(grid_search.cv_results_)
df = df.sort_values("rank_test_score")
df.to_csv("csv_result.csv")
