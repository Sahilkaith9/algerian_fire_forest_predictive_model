from flask import Flask,render_template,url_for,request
import pickle


application=Flask(__name__)
app=application

ridge_model=pickle.load(open('models/project.pkl','rb'))
scaler=pickle.load(open('models/scaler.pkl','rb'))


@app.route("/")
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict_data():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=request.form.get('FFMC')
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=request.form.get('Classes')
        Region=request.form.get('Region')
        scaled_value=scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes,Region]])
        result=ridge_model.predict(scaled_value)
        return render_template('predict.html',results=result)
    else:
        return render_template('predict.html')


if __name__=="__main__":
    app.run(debug=True)
