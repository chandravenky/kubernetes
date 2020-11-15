import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('classifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    Render results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = prediction[0]
    
    return render_template('index.html', prediction_val='The model predicts the risk to be a {}'.format(output))

@app.route('/predict_api',methods=['GET'])
def predict_api():
    '''
    Direct API calls
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    
    output = prediction[0]
    
    return jsonify(output)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 5000, debug=True)