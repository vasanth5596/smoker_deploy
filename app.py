import pandas as pd 
import numpy as  np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    if output == 0:
        return render_template('index.html', prediction_text=' your are not a Smoker')
    else:
        return render_template('index.html', prediction_text='Smoking is injurious to health ...you are a Smoker !!!')
    



# if __name__ == "__main__":
#     app.run(debug=True)
    
    
    