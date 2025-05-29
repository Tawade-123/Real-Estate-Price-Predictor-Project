from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('random.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    bhk = int(request.form['bhk'])
    type_ = int(request.form['type'])
    area = int(request.form['area'])
    status = int(request.form['status'])
    age = int(request.form['age'])

    # Prepare the input array
    input_features = np.array([[bhk, type_, area, status, age]])

    # Predict the price
    prediction = model.predict(input_features)[0]

    return render_template('output.html', prediction_text=f'Predicted Price: {prediction} Lakhs')


if __name__ == '__main__':
    app.run(debug=True)
