import sys
print(sys.executable)

from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model

# Create flask app
flask_app = Flask(__name__)
model = load_model('model.h5')

@flask_app.route("/")
def Home():
    return render_template("home.html", form_data={})

@flask_app.route("/predict", methods=["POST"])
def predict():
    # Define all conditions
    conditions = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']

    # Extract features from request form
    form_data = request.form

    # Extract age and gender
    age = form_data.get('age')
    gender = form_data.get('gender')
    if age == '':
       age = 48
    if gender in ['0', '', None]:
       gender = 'male'

    # Extract height and weight
    height_feet = form_data.get('height-feet')
    if height_feet:
        heightF = float(height_feet)
    else:
        heightF = 1.0
    height_inches = form_data.get('height-inches')
    if height_inches:
        heightI = float(height_inches)
    else:
        heightI = 1.0
    weight_str = form_data.get('weight')
    if weight_str:
        weight = float(weight_str)
    else:
        weight = 1.0

    #calculate bmi
    inches = heightI + heightF * 12
    bmi = (weight / inches **2) * 703
    print(f"User bmi: {bmi}")

    # Check if "Obesity" is 0 and if bmi is greater than or equal to 30
    form_data = form_data.to_dict()  # Convert ImmutableMultiDict to dict

    obesity_value = form_data.get('Obesity')
    if obesity_value in ['0', '', None] and bmi >= 30:
        form_data['Obesity'] = '1'

    # Print age and gender
    print(f"User selected age: {age}")
    print(f"User selected gender: {gender}")

    # Encode gender
    gender = 1 if gender == 'male' else 0

    # Encode conditions
    encoded_conditions = [1 if form_data.get(condition) else 0 for condition in conditions]

    # Print conditions
    for condition, encoded in zip(conditions, encoded_conditions):
        if encoded:
            print(f"User selected condition: {condition}")

    # Combine all features
    features = [age, gender] + encoded_conditions

    # Convert features to numerical array
    float_features = [float(x) if x else 0.0 for x in features]
    features = np.array(float_features).reshape(1, -1)

    # Make prediction using the model
    prediction_proba = model.predict(features)

    # Convert prediction to a readable format (if necessary)
    if prediction_proba[0][0] > 0.5:
        prediction_text = "Positive"
    else:
        prediction_text = "Negative"

    predictionProb = "{:.2f}".format(prediction_proba[0][0])

    # Pass the prediction, raw prediction value, and form data back to the template
    return render_template("home.html", prediction=prediction_text, prediction_prob=predictionProb, prediction_value=prediction_proba[0][0], form_data=form_data)

if __name__ == "__main__":
    flask_app.run(debug=True)
