import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_features = pd.read_csv("diabetes_data_upload.csv")

# Separate features and labels
data_labels = data_features.loc[:, ["class"]]
data_features = data_features.drop("class", axis=1)

# Encoding categorical features
data_features_encoded = pd.get_dummies(data_features, dtype=int)

# Encoding categorical labels
data_labels_encoded = pd.get_dummies(data_labels, dtype=int)

#  z-score standardization scaler x = (x - u) / s
scaler = StandardScaler()
x_train = scaler.fit_transform(data_features_encoded)

# Convert labels to numpy array
y_train = data_labels_encoded.to_numpy(dtype='float32')

# Convert numpy arrays to DataFrames
x_train_df = pd.DataFrame(x_train, columns=data_features_encoded.columns)
y_train_df = pd.DataFrame(y_train, columns=data_labels_encoded.columns)

# Save DataFrames to CSV files
x_train_df.to_csv("x_train.csv", index=False)
y_train_df.to_csv("y_train.csv", index=False)
