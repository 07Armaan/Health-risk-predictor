from flask import Flask,render_template,request
from ouliers import outliers_handler
import joblib
import pandas as pd

app = Flask(__name__)
final_pipeline = joblib.load("final_pipeline.pkl")
le = joblib.load("le.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def prediction():
    age = request.form.get("age")
    weight = request.form.get("weight")
    height = request.form.get("height")
    exercise = request.form.get("exercise")
    sleep = request.form.get("sleep")
    sugar_intake = request.form.get("sugar_intake")
    smoking = request.form.get("smoking")
    alcohol = request.form.get("alcohol")
    married = request.form.get("married")
    profession = request.form.get("profession")
    bmi = request.form.get("bmi")
    features = [[age,weight,height,exercise,sleep,sugar_intake,smoking,alcohol,married,profession,bmi]]
    feature_names = ["age","weight","height","exercise","sleep","sugar_intake","smoking","alcohol","married","profession","bmi"]
    test_df = pd.DataFrame(features,columns=feature_names)

    features_pred = final_pipeline.predict(test_df)
    pred = le.inverse_transform(features_pred)[0]
    return  render_template("index.html",prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)