import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
diabetes = pd.read_csv('diabetes_data_upload.csv')
from sklearn.preprocessing import LabelEncoder

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

plt.figure(figsize=(10,8))
sns.heatmap(diabetes.corr(), annot=True, cmap='coolwarm')
plt.show()