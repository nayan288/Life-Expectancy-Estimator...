from flask import Flask, request, render_template
import joblib

model = joblib.load("trainedmodel.joblib")

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def death():
    if request.method == "POST":
        Age = request.form.get("Age")
        Sex = request.form.get("Sex")
        Development_Level = request.form.get("Development_Level")
        Urban_Rural = request.form.get("Urban_Rural")
        Education_Level = request.form.get("Education_Level")
        Income_Bracket = request.form.get("Income_Bracket")
        Marital_Status = request.form.get("Marital_Status")
        Employment_Status = request.form.get("Employment_Status")
        Occupation_Risk = request.form.get("Occupation_Risk")
        Height_cm = request.form.get("Height_cm")
        Weight_kg = request.form.get("Weight_kg")
        BMI = request.form.get("BMI")
        Waist_cm = request.form.get("Waist_cm")
        Systolic_BP = request.form.get("Systolic_BP")
        Diastolic_BP = request.form.get("Diastolic_BP")
        Resting_Heart_Rate = request.form.get("Resting_Heart_Rate")
        Total_Cholesterol = request.form.get("Total_Cholesterol")
        HDL_Cholesterol = request.form.get("HDL_Cholesterol")
        Fasting_Glucose = request.form.get("Fasting_Glucose")
        Smoking_Status = request.form.get("Smoking_Status")
        Cigarettes_Per_Day = request.form.get("Cigarettes_Per_Day")
        Years_Since_Quit = request.form.get("Years_Since_Quit")
        Alcohol_Drinks_Per_Week = request.form.get("Alcohol_Drinks_Per_Week")
        Exercise_Min_Per_Week = request.form.get("Exercise_Min_Per_Week")
        Steps_Per_Day = request.form.get("Steps_Per_Day")
        Diet_Quality = request.form.get("Diet_Quality")
        Fruit_Veg_Servings_Per_Day = request.form.get("Fruit_Veg_Servings_Per_Day")
        Sugary_Drinks_Per_Week = request.form.get("Sugary_Drinks_Per_Week")
        Sleep_Hours = request.form.get("Sleep_Hours")
        Sleep_Quality = request.form.get("Sleep_Quality")
        Sitting_Hours_Per_Day = request.form.get("Sitting_Hours_Per_Day")
        Drug_Use = request.form.get("Drug_Use")
        Driving_Habits = request.form.get("Driving_Habits")
        Diabetes_Status = request.form.get("Diabetes_Status")
        Hypertension = request.form.get("Hypertension")
        Heart_Disease = request.form.get("Heart_Disease")
        Stroke_History = request.form.get("Stroke_History")
        Cancer_History = request.form.get("Cancer_History")
        Chronic_Lung_Disease = request.form.get("Chronic_Lung_Disease")
        Kidney_Disease = request.form.get("Kidney_Disease")
        Depression_Anxiety = request.form.get("Depression_Anxiety")
        Number_Of_Medications = request.form.get("Number_Of_Medications")
        Hospitalizations_Last_5Yrs = request.form.get("Hospitalizations_Last_5Yrs")
        Annual_Checkup = request.form.get("Annual_Checkup")
        Vaccinated = request.form.get("Vaccinated")
        Parents_Lived_To_80 = request.form.get("Parents_Lived_To_80")
        Family_Heart_Disease = request.form.get("Family_Heart_Disease")
        Family_Cancer = request.form.get("Family_Cancer")
        Social_Connection = request.form.get("Social_Connection")
        Stress_Level = request.form.get("Stress_Level")
        Life_Satisfaction = request.form.get("Life_Satisfaction")
        Owns_Pet = request.form.get("Owns_Pet")
        Air_Pollution_PM25 = request.form.get("Air_Pollution_PM25")
        Healthcare_Access = request.form.get("Healthcare_Access")

        data = [[Age, Sex, Development_Level, Urban_Rural, Education_Level, Income_Bracket, Marital_Status, Employment_Status, Occupation_Risk, Height_cm, Weight_kg, BMI, Waist_cm, Systolic_BP, Diastolic_BP, Resting_Heart_Rate, Total_Cholesterol, HDL_Cholesterol, Fasting_Glucose, Smoking_Status, Cigarettes_Per_Day, Years_Since_Quit, Alcohol_Drinks_Per_Week, Exercise_Min_Per_Week, Steps_Per_Day, Diet_Quality, Fruit_Veg_Servings_Per_Day, Sugary_Drinks_Per_Week, Sleep_Hours, Sleep_Quality, Sitting_Hours_Per_Day, Drug_Use, Driving_Habits, Diabetes_Status, Hypertension, Heart_Disease, Stroke_History, Cancer_History, Chronic_Lung_Disease, Kidney_Disease, Depression_Anxiety, Number_Of_Medications, Hospitalizations_Last_5Yrs, Annual_Checkup, Vaccinated, Parents_Lived_To_80, Family_Heart_Disease, Family_Cancer, Social_Connection, Stress_Level, Life_Satisfaction, Owns_Pet, Air_Pollution_PM25, Healthcare_Access]]
        Years_Remaining = model.predict(data)

        Year = Years_Remaining[0]

        return render_template("index.html",result= f"Your Remaining Year is {Year}")
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0")