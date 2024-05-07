import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree

# data preprocessing
data_features = pd.read_csv("..\\diabetes_data_upload.csv")
data_labels = data_features.loc[:, ["class"]]
data_features = data_features.drop("class", axis=1)

# encoding categorical value with numerical or boolean value
data_features_encoded = pd.get_dummies(data_features, drop_first=True, dtype=int)
data_labels_encoded = pd.get_dummies(data_labels, drop_first=True, dtype=int)

x_train, x_test, y_train, y_test = train_test_split(data_features_encoded, data_labels_encoded, test_size=0.2)

best_params = ""
best_f1 = 0
best_acc = 0
best_pre = 0
best_rec = 0
dtree = any

# hyperparameters tuning
for criterion in ["gini", "entropy"]:
    for max_depth in range(2, 15):
        for min_samples_leaf in range(1, 20):
            dtree = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, min_samples_leaf=min_samples_leaf)
            dtree.fit(x_train, y_train)
            test_prediction = dtree.predict(x_test)
            test_acc = accuracy_score(y_test, test_prediction)
            test_precision = precision_score(y_test, test_prediction)
            test_recall = recall_score(y_test, test_prediction)
            test_f1 = f1_score(y_test, test_prediction)
            if test_f1 > best_f1:
                best_params = f"criterion: {criterion}, max_depth: {max_depth}, min_samples_leaf: {min_samples_leaf}"
                best_f1 = test_f1
                best_acc = test_acc
                best_pre = test_precision
                best_rec = test_recall

print(best_params)
print(best_acc)
print(best_pre)
print(best_rec)
print(best_f1)

fig = plt.figure(figsize=(50, 50))
plot_tree(dtree,
          feature_names=data_features_encoded.columns,
          class_names=['no diabetes', 'diabetes'],
          impurity=False,
          proportion=True,
          filled=True)
fig.savefig("Diabetes Decision Tree.png")
