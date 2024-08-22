from flask import Flask,request,jsonify
from flask_cors import CORS
import pickle
import numpy as np

app=Flask(__name__)
CORS(app,resources={r"/predict": {"origins": "http://localhost:3000"}})
with open('model/car_price_predictor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    

@app.route('/predict', methods=['POST'])
def predict():
    print(request)
    data = request.json    
    mileage =  float(data.get('mileage')) if data.get('mileage') else 19.3
    km_driven=float(data.get('km_driven')) if data.get('km_driven') else 60000.0
    year = float(data.get('year')) if data.get('year') else 2015
    
    if mileage is None and year is None and km_driven is None:
        return jsonify({'error': 'Please provide mileage, year, and km_Driven'}), 400

    
    
    # # Prepare the data for prediction
    features = np.array([[km_driven,mileage, year ]])
    
    # # Perform the prediction using the loaded model
    prediction = model.predict(features)
    return jsonify({'predicted_price': round(np.exp(prediction[0]),2)})

@app.route('/', methods=['GET'])
def call():
    return jsonify({'Name':"Suryansh Srivastava"})
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)