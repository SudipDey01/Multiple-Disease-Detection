from telnetlib import BM
from flask import Flask,render_template,url_for,request
from flask_material import Material
import pandas as pd 
import numpy as np 

# ML Pkg
import joblib


app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/diabetes')
def diabetes():
    return render_template("diabetes.html")

@app.route('/heart')
def heart():
    return render_template("heart.html")

@app.route('/parkinsons')
def parkinsons():
    return render_template("parkinsons.html")



@app.route('/diabetes.html',methods=["POST"])
def diabetes_analyze():
	if request.method == 'POST':
		Pregnancies = request.form['Pregnancies']
		Glucose = request.form['Glucose']
		BloodPressure = request.form['BloodPressure']
		SkinThickness = request.form['SkinThickness']
		Insulin = request.form['Insulin']
		Bmi = request.form['Bmi']
		DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
		Age = request.form['Age']

		# Clean the data by convert from unicode to float 
		sample_data = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,Bmi,DiabetesPedigreeFunction,Age]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# Reloading the Model
		diabetes_model = joblib.load('data/diabetes_model.pkl')
		result_prediction = diabetes_model.predict(ex1)

	return render_template('diabetes.html', Pregnancies=Pregnancies,
		Glucose=Glucose,
		BloodPressure=BloodPressure,
		SkinThickness=SkinThickness,
		Insulin=Insulin,
		Bmi=Bmi,
		DiabetesPedigreeFunction=DiabetesPedigreeFunction,
		Age=Age,
		clean_data=clean_data,
		result_prediction=result_prediction)

@app.route('/heart.html',methods=["POST"])
def heart_analyze():
	if request.method == 'POST':
		cp= request.form['cp']
		trestbps= request.form['trestbps']
		chol= request.form['chol']
		fbs= request.form['fbs']
		restecg= request.form['restecg']
		thalach= request.form['thalach']
		exang= request.form['exang']

		# Clean the data by convert from unicode to float 
		sample_data =[cp, trestbps,chol,fbs,restecg,thalach,exang]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# Reloading the Model
		heart_model = joblib.load('data/heart_model.pkl')
		result_prediction = heart_model.predict(ex1)

	return render_template('heart.html', cp=cp,
		trestbps=trestbps,
		chol=chol,
		fbs=fbs,
		restecg=restecg,
		thalach=thalach,
		exang=exang,
		clean_data=clean_data,
		result_prediction=result_prediction)





if __name__ == '__main__':
	app.run(debug=True)