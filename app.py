from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model from the pickle file
model = pickle.load(open('C:/Users/acer/Breast_Cancer/dataset/model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Extract patient information and symptoms from form
        patient_details = {
            'patient_name': request.form['patient_name'],
            'age': request.form['age'],
            'sex': request.form['sex'],
            'blood_group': request.form['blood_group'],
            'radius_mean': float(request.form['radius_mean']),
            'texture_mean': float(request.form['texture_mean']),
            'perimeter_mean': float(request.form['perimeter_mean']),
            'concavity_mean': float(request.form['concavity_mean']),
            'concave_points_mean': float(request.form['concave_points_mean']),
            'radius_worst': float(request.form['radius_worst']),
            'texture_worst': float(request.form['texture_worst']),
            'perimeter_worst': float(request.form['perimeter_worst']),
            'smoothness_worst': float(request.form['smoothness_worst']),
            'concavity_worst': float(request.form['concavity_worst']),
            'concave_points_worst': float(request.form['concave_points_worst']),
            'symmetry_worst': float(request.form['symmetry_worst']),
            'fractal_dimension_worst': float(request.form['fractal_dimension_worst'])
        }

        # List of features for the model, ensure they are in the correct order
        features = [
            patient_details['radius_mean'],
            patient_details['texture_mean'],
            patient_details['perimeter_mean'],
            patient_details['concavity_mean'],
            patient_details['concave_points_mean'],
            patient_details['radius_worst'],
            patient_details['texture_worst'],
            patient_details['perimeter_worst'],
            patient_details['smoothness_worst'],
            patient_details['concavity_worst'],
            patient_details['concave_points_worst'],
            patient_details['symmetry_worst'],
            patient_details['fractal_dimension_worst'],
        ]

        # Convert features into numpy array and reshape for the model input
        features_array = np.array(features).reshape(1, -1)

        # Make prediction using the loaded model
        prediction = model.predict(features_array)[0]
        result = "Malignant" if prediction == 1 else "Benign"

        # Redirect to the prediction page with results and patient details
        return render_template('prediction.html', details=patient_details, result=result)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False)
