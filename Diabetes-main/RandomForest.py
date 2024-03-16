from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import GridSearchCV



# Load the dataset
diabetes = pd.read_csv('diabetes_data_upload.csv')
le = LabelEncoder()

diabetes['Gender'] = le.fit_transform(diabetes['Gender'])
diabetes['Polyuria'] = le.fit_transform(diabetes['Polyuria'])
diabetes['Polydipsia'] = le.fit_transform(diabetes['Polydipsia'])
diabetes['sudden weight loss'] = le.fit_transform(diabetes['sudden weight loss'])
diabetes['weakness'] = le.fit_transform(diabetes['weakness'])
diabetes['Polyphagia'] = le.fit_transform(diabetes['Polyphagia'])
diabetes['Genital thrush'] = le.fit_transform(diabetes['Genital thrush'])
diabetes['visual blurring'] = le.fit_transform(diabetes['visual blurring'])
diabetes['Itching'] = le.fit_transform(diabetes['Itching'])
diabetes['Irritability'] = le.fit_transform(diabetes['Irritability'])
diabetes['delayed healing'] = le.fit_transform(diabetes['delayed healing'])
diabetes['partial paresis'] = le.fit_transform(diabetes['partial paresis'])
diabetes['muscle stiffness'] = le.fit_transform(diabetes['muscle stiffness'])
diabetes['Alopecia'] = le.fit_transform(diabetes['Alopecia'])
diabetes['Obesity'] = le.fit_transform(diabetes['Obesity'])
diabetes['class'] = le.fit_transform(diabetes['class'])

# Assume the last column is the target and the rest are features
X = diabetes.iloc[:, :-1]
y = diabetes.iloc[:, -1]

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=64, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Check the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')
print(classification_report(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

n_estimators_range = list(range(1, 101))
param_grid = dict(n_estimators=n_estimators_range)
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)

# View the complete results
grid.cv_results_

# Examine the best model
print(grid.best_score_)
print(grid.best_params_)    
