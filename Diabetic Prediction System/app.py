from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import numpy as np
import pandas as pd
import joblib
import logging
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.logger.setLevel(logging.DEBUG)

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load the trained ML model
try:
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("Model files not found! Please train the model first by running train_model.py")
    exit()

def create_database():
    with sqlite3.connect("user.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                password TEXT,
                prediction_result TEXT
            )
        ''')
        conn.commit()

create_database()


@app.route('/')
def home():
    return render_template("home.html")  # Public homepage before login

@app.route('/signup', methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        app.logger.debug(f"Signup attempt for username: {username}")

        if not username or not password:
            flash("Username and password cannot be empty!", "error")
            return redirect(url_for("signup"))

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password)

        try:
            with sqlite3.connect("user.db") as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (name, password, prediction_result) VALUES (?, ?, ?)", 
                               (username, hashed_password, ""))
                conn.commit()

            app.logger.debug(f"User {username} created successfully")
            flash("Signup successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "error")
            return redirect(url_for("signup"))

    return render_template("signup.html")

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        app.logger.debug(f"Login attempt for username: {username}")

        if not username or not password:
            flash("Username and password cannot be empty!", "error")
            return redirect(url_for("login"))

        with sqlite3.connect("user.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE name=?", (username,))
            user = cursor.fetchone()

        if user and check_password_hash(user[2], password):
            session["user"] = username
            app.logger.debug(f"Redirecting to main for user: {username}")
            return redirect(url_for("main"))
        else:
            app.logger.debug(f"Login failed for username: {username}")
            flash("Invalid username or password!", "error")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route('/main')
def main():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("main.html", user=session["user"])

# Prediction Page
@app.route('/info')
def info():
    return render_template("info.html")

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        print("Received Form Data:", request.form.to_dict())  # Debugging print statement

        required_fields = ["glucose", "blood_pressure", "bmi", "skin_thickness", "insulin", "pregnancies", "age", "diabetes_pedigree"]
        missing_fields = [field for field in required_fields if field not in request.form]

        if missing_fields:
            flash(f"Missing fields: {', '.join(missing_fields)}", "error")
            return redirect(url_for("predict"))

        try:
            # Create DataFrame with CORRECT CAPITALIZED column names matching dataset/train_model.py
            feature_data = [
                float(request.form["pregnancies"]),
                float(request.form["glucose"]),
                float(request.form["blood_pressure"]),
                float(request.form["skin_thickness"]),
                float(request.form["insulin"]),
                float(request.form["bmi"]),
                float(request.form["diabetes_pedigree"]),
                float(request.form["age"])
            ]
            app.logger.debug(f"Input features received for prediction: {feature_data}")
            
            features_df = pd.DataFrame([feature_data], columns=[
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ])
            
            # Scale and predict
            features_scaled = scaler.transform(features_df)
            prediction = model.predict(features_scaled)
            result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
            app.logger.debug(f"Prediction made: {result} for features: {features_scaled}")

            # Save to DB
            with sqlite3.connect("user.db") as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET prediction_result=? WHERE name=?", (result, session["user"]))
                conn.commit()
            
            app.logger.debug(f"Prediction result stored for user {session['user']}: {result}")
            return render_template("result.html", prediction=result)

        except Exception as e:
            app.logger.error(f"Error during prediction: {str(e)}")
            flash(f"Prediction error: {str(e)}", "error")
            return redirect(url_for("predict"))

    return render_template("predict.html")  # Render form for GET requests

# Logout
@app.route('/logout')
def logout():
    session.pop("user", None)
    flash("Logged out successfully!", "success")
    return redirect(url_for("login"))

# BMI Calculation
@app.route('/bmi', methods=['GET', 'POST'])
def bmi():
    if request.method == 'POST':
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        bmi_value = weight / (height ** 2)
        return render_template('bmi.html', bmi=bmi_value)
    return render_template('bmi.html')

# Skin Thickness Estimation
@app.route('/skin_thickness', methods=['GET', 'POST'])
def skin_thickness():
    if request.method == 'POST':
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        bmi = weight / (height ** 2)
        skin_thickness_value = 0.5 * bmi + 5
        return render_template('skin_thickness.html', skin_thickness=skin_thickness_value)
    return render_template('skin_thickness.html')

# Diabetes Pedigree Function Calculation Route
@app.route('/diabetes_pedigree', methods=['GET', 'POST'])
def diabetes_pedigree():
    if request.method == 'POST':
        try:
            mother = int(request.form.get('mother', 0))
            father = int(request.form.get('father', 0))
            siblings = int(request.form.get('siblings', 0))

            pedigree_score = (mother * 0.5 + father * 0.5 + siblings * 0.25) / 3

            return render_template('diabetes_pedigree.html', pedigree_score=pedigree_score)

        except ValueError:
            flash("Please enter valid numbers for family history!", "error")
            return redirect(url_for("diabetes_pedigree"))

    return render_template('diabetes_pedigree.html')


if __name__ == "__main__":
    app.run(debug=True)