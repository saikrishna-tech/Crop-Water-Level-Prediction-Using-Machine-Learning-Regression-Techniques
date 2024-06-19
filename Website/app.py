import sys
import os
import glob
import re

import numpy as np
import pandas as pd
import math
import pickle

from PIL import Image

from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template

import matplotlib.pyplot as plt
from matplotlib.image import imread


app = Flask(__name__)


################################ All pages server connection ##########################################

@app.route("/", methods=["GET", "POST"]) 
def runhome():
    return render_template("index.html") 

@app.route("/lc_dl", methods=["GET", "POST"]) 
def lc_dl():
    return render_template("lung_cancer_classification.html")

################################--- lung cancer classification ----############################################################################

decision_tree_model = pickle.load(open('decision_tree_model.pkl', 'rb'))

@app.route('/predict_crop_water', methods=['GET', 'POST'])
def predict_crop_water():
    if request.method == "POST":
        # Get form data
        ct = request.form['crop_type']
        st = request.form['soil_type']
        re = request.form['region']
        tp = request.form['temperature']
        wh = request.form['weather_condition']

        # Mapping of crop types, soil types, regions, temperature ranges, and weather conditions
        c_t = ['BANANA', 'BEAN', 'CABBAGE', 'CITRUS', 'COTTON', 'MAIZE', 'MELON', 'MUSTARD', 'ONION', 'POTATO', 'RICE', 'SOYABEAN', 'SUGARCANE', 'TOMATO', 'WHEAT']
        s_t = ['DRY', 'HUMID', 'WET']
        r_e = ['DESERT', 'HUMID', 'SEMI ARID', 'SEMI HUMID']
        t_p = ['20-30', '30-40', '40-50', '10-20']
        w_h = ['NORMAL', 'RAINY', 'SUNNY', 'WINDY']

        # Get indices of selected options
        ct1 = c_t.index(ct.upper())
        st1 = s_t.index(st.upper())
        re1 = r_e.index(re.upper())
        tp1 = t_p.index(tp.upper())
        wh1 = w_h.index(wh.upper())

        # Create input array for prediction
        arr = np.array([ct1, st1, re1, tp1, wh1]).reshape(1, -1)

        # Make prediction
        prediction = decision_tree_model.predict(arr)

        # Prepare response
        predicted_irrigation_level = prediction[0]

        # Pass inputs and prediction result to the results page
        return render_template("result.html", 
            crop_type=ct, soil_type=st, region=re, temperature=tp, weather_condition=wh,
            predicted_irrigation_level=predicted_irrigation_level)
    else:
        return "Sorry!!"


################################################################################################


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
