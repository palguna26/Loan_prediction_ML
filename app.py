from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('loan_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        features = [
            int(data['Gender']),
            int(data['Married']),
            int(data['Dependents']),
            int(data['Education']),
            int(data['Self_Employed']),
            float(data['ApplicantIncome']),
            float(data['CoapplicantIncome']),
            float(data['LoanAmount']),
            float(data['Loan_Amount_Term']),
            float(data['Credit_History']),
            int(data['Property_Area']),
        ]
        pred = model.predict([features])[0]
        result = "✅ Eligible for Loan" if pred == 1 else "❌ Not Eligible for Loan"
        return render_template('result.html', prediction=result)
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
