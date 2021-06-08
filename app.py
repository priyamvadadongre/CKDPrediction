import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
import pickle
#from ckd_prediction_ import sc

app = Flask(__name__)# app creation
model = pickle.load(open('lgmodel.pkl', 'rb'))# lloading pkl
scalr = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():                                #home page 
    return render_template('index.html')
 

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    '''        
    means=[ 0.00000000e+00, -2.30926389e-16,  3.18323146e-14,  3.55271368e-17,
       -7.10542736e-17,  7.10542736e-17,  0.00000000e+00,  1.77635684e-17,
       -5.32907052e-17,  0.00000000e+00, -1.06581410e-16,  7.10542736e-17,
        6.03961325e-16, -2.13162821e-16, -1.77635684e-17,  3.55271368e-16,
        3.55271368e-17, -7.10542736e-17, -9.76996262e-17,  0.00000000e+00,
       -1.77635684e-17,  0.00000000e+00]
    varia=[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1.]
    def ScaleData(ar,mea=means,va=varia):
        for i in range(0,22):
            ar[i]=((ar[i]-mea[i])/(va[i])) 
        return ar  '''
    int_features = [[float(x) for x in request.form.values()]]
    final_fea=scalr.transform(int_features)
   
   
    
    #features=sc.transform(int_features)
    prediction = model.predict(final_fea)
    if(prediction[0]==0.):
        output='DOESNT HAVE CKD'
    else:
        output=' HAS CKD'
    return render_template('index.html', prediction_text='THE PATIENT  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)