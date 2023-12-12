from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict')
def predict():
    try:
        # Get input values from the POST request
        cgpa = float(request.form.get('cgpa'))
        iq = float(request.form.get('iq'))
        profile_score = float(request.form.get('profile_score'))

        # Create a NumPy array with the input values
        input_query = np.array([[cgpa, iq, profile_score]])

        # Make predictions
        result = model.predict(input_query)[0]

        # Return the result as JSON
        return jsonify({'placement': str(result)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
