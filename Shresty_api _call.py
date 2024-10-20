from flask import Flask, request, jsonify
import pandas as pd
import joblib  

#Import joblib to load the saved model

app = Flask(__name__)


# Load the trained model from the file
model = joblib.load('loan_model.pkl')  

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Loan Prediction API. Use /predict with POST to make predictions."


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    user_data = pd.DataFrame([data])

    prediction = model.predict(user_data)[0]  # Assumes binary classification (1 or 0)

    result = 'approved' if prediction == 1 else 'rejected'

    return jsonify({'loan_status': result})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
