from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Loadtrained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the HTML form
    input1 = float(request.form['input1'])
    input2 = float(request.form['input2'])
    input3 = float(request.form['input3'])

    # Make a prediction using the loaded model
    inputs = np.array([[input1, input2, input3]])
    prediction = model.predict(inputs)

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
