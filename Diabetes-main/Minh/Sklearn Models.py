import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt

# data preprocessing
data_features = pd.read_csv("..\\diabetes_data_upload.csv")
data_labels = data_features.loc[:, ["class"]]
data_features = data_features.drop("class", axis=1)

# encoding categorical value with numerical or boolean value
data_features_encoded = pd.get_dummies(data_features, drop_first=True, dtype=float)
data_labels_encoded = pd.get_dummies(data_labels, drop_first=True, dtype=float)

x_values = data_features_encoded.values
y_values = data_labels_encoded.values.flatten()

x_mean = x_values.mean(axis=0)
x_std = x_values.std(axis=0)

# z-scale transformation
x_normal = (x_values - x_mean) / x_std

x_train, x_test, y_train, y_test = train_test_split(
    x_normal, y_values, test_size=0.2, random_state=1, stratify=y_values)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=1, stratify=y_train)


models = {'Sklearn Logistic Regression': LogisticRegression(), 'Sklearn Support Vector Machines': LinearSVC(dual=True),
          'Sklearn Random Forest': RandomForestClassifier(), 'Sklearn Naive Bayes': GaussianNB(),
          'Sklearn K-Nearest Neighbor': KNeighborsClassifier(), 'Sklearn Decision Trees': DecisionTreeClassifier()}

accuracy, precision, recall = {}, {}, {}

for key in models.keys():
    # Fit the classifier
    models[key].fit(x_train, y_train)

    # Make predictions
    predictions = models[key].predict(x_test)

    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)

models['Logistic Regression'] = 0
accuracy['Logistic Regression'] = 0.9615
precision['Logistic Regression'] = 0
recall['Logistic Regression'] = 0

models['MLP'] = 0
accuracy['MLP'] = 0.9712
precision['MLP'] = 0
recall['MLP'] = 0

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()

df_model = df_model.sort_values(by=['Accuracy'])

print(df_model)

ax = df_model.plot.barh()
ax.legend(
    ncol=len(models.keys()),
    bbox_to_anchor=(0, 1),
    loc='lower left',
    prop={'size': 14}
)
plt.tight_layout()
plt.show()
