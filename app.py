from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'Jackjeevaadmin'  # Important for sessions!

# Load model, scaler, and features
model = joblib.load('ransomware_model_15.pkl')
scaler = joblib.load('scaler_15.pkl')
top_features = joblib.load('top_features.pkl')

# Hardcoded admin credentials
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'jack'

@app.route('/')
def home():
    if 'logged_in' in session and session['logged_in']:
        return render_template('index.html', features=top_features)
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            error = 'Invalid Credentials. Please try again.'
            return render_template('login.html', error=error)
    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))
    
    try:
        input_data = [float(request.form[feature]) for feature in top_features]
        input_array = np.array(input_data).reshape(1, -1)

        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)

        result = 'Ransomware' if prediction[0] == 1 else 'Benign'

        return render_template('index.html', features=top_features, prediction=result)
    except Exception as e:
        return f"Error: {e}"

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
