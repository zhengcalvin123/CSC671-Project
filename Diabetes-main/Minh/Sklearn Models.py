import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

# non-negative dataset
xx_train, xx_test, yy_train, yy_test = train_test_split(
    x_values, y_values, test_size=0.2, random_state=1, stratify=y_values)

models = {'Sklearn Logistic Regression': LogisticRegression(),
          'Sklearn Support Vector Machines': LinearSVC(dual=True),
          'Sklearn Random Forest': RandomForestClassifier(),
          'Sklearn GaussianNB': GaussianNB(),
          'Sklearn BernoulliNB': BernoulliNB(),
          'Sklearn K-Nearest Neighbor': KNeighborsClassifier(),
          'Sklearn Decision Trees': DecisionTreeClassifier()}

accuracy, precision, recall, f1 = {}, {}, {}, {}

for key in models.keys():
    # Fit the classifier
    models[key].fit(x_train, y_train)

    # Make predictions
    predictions = models[key].predict(x_test)

    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)
    f1[key] = f1_score(predictions, y_test)

model = MultinomialNB()
model.fit(xx_train, yy_train)
predictions = model.predict(xx_test)
accuracy_multinomialNB = accuracy_score(predictions, yy_test)
precision_multinomialNB = precision_score(predictions, yy_test)
recall_multinomialNB = recall_score(predictions, yy_test)
f1_multinomialNB = f1_score(predictions, yy_test)

models['SkLearn MultinomialNB'] = 0
accuracy['Sklearn MultinomialNB'] = accuracy_multinomialNB
precision['Sklearn MultinomialNB'] = precision_multinomialNB
recall['Sklearn MultinomialNB'] = recall_multinomialNB
f1['Sklearn MultinomialNB'] = f1_multinomialNB

models['Manual Sklearn Decision Trees'] = 0
accuracy['Manual Sklearn Decision Trees'] = 0.9711
precision['Manual Sklearn Decision Trees'] = 0.9531
recall['Manual Sklearn Decision Trees'] = 1.000
f1['Manual Sklearn Decision Trees'] = 0.9760

models['Logistic Regression'] = 0
accuracy['Logistic Regression'] = 0.9615
precision['Logistic Regression'] = 0.6100
recall['Logistic Regression'] = 0.9531
f1['Logistic Regression'] = 0.7439

models['MLP'] = 0
accuracy['MLP'] = 0.9712
precision['MLP'] = 0.6139
recall['MLP'] = 0.9688
f1['MLP'] = 0.7515

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()
df_model['F1-Score'] = f1.values()

df_model = df_model.sort_values(by=['Accuracy'], ascending=True)

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
