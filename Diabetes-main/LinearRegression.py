import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error


# Load data
diabetes = pd.read_csv('diabetes_data_upload.csv')

# Create a label encoder object
le = LabelEncoder()

# Fit and transform the columns into numerical values
for col in diabetes.columns:
    diabetes[col] = le.fit_transform(diabetes[col])

diabetes_features = diabetes.copy()
diabetes_labels = diabetes_features.pop('class')

# Split data into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(diabetes_features, diabetes_labels, test_size=0.2)

# Define model with optimal hyperparameters
diabetes_model = tf.keras.Sequential([
  layers.Dense(32, activation='sigmoid'),
  layers.Dense(1, activation='sigmoid')
])

# Compile model with optimal learning rate
diabetes_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), metrics=['accuracy'])

# Train model with optimal number of epochs
features_train = features_train.astype('float32')
labels_train = labels_train.astype('float32')
features_test = features_test.astype('float32')
labels_test = labels_test.astype('float32')
history = diabetes_model.fit(features_train, labels_train, epochs=1000, validation_data=(features_test, labels_test))

# Evaluate model
test_loss, test_acc = diabetes_model.evaluate(features_test, labels_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
X = diabetes.drop('class', axis = 1)
y = diabetes['class']

scaler = MinMaxScaler(feature_range=(0, 1))
X_rescaled = scaler.fit_transform(X)
X = pd.DataFrame(data = X_rescaled, columns = X.columns)

set_of_classes = y.value_counts().index.tolist()
set_of_classes= pd.DataFrame({'class': set_of_classes})
y = pd.get_dummies(y)

max_iterations = [500, 1000, 1500, 2000]

hidden_layer_siz = [(32,), (64,), (128,)]
learning_rates = [0.05, 0.1, 0.15]

param_grid = dict(learning_rate_init = learning_rates, hidden_layer_sizes = hidden_layer_siz, max_iter = max_iterations)

# set model
mlp = MLPClassifier(solver = 'sgd', random_state = 42, activation = 'logistic', learning_rate_init = 0.3, batch_size = 100, hidden_layer_sizes = (64,), max_iter = 1000)

# For Grid Search
grid = GridSearchCV(estimator = mlp, param_grid = param_grid)

grid.fit(X,y)

y_pred = grid.predict(X)
mse = mean_squared_error(y, y_pred)
print("Optimal Hyper-parameters : ", grid.best_params_)
print("Optimal Accuracy : ", grid.best_score_)
print("Minimum MSE: ", mse)


diabetes_model.save('model.h5')
