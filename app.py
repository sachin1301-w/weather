import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import os

API_KEY = '8db8619928b803f9636ab88525b1e0ba'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

app = Flask(__name__)

# ---------- Data Functions ----------
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code != 200 or "main" not in data:
        return None
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'WindGustSpeed': data['wind']['speed']
    }

def read_historical_data(filename="weather.csv"):
    df = pd.read_csv(filename)
    df = df.dropna().drop_duplicates()
    return df

def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
    X = data[['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp']]
    y = data['RainTomorrow']
    return X, y, le

def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data)-1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])
    return np.array(X).reshape(-1,1), np.array(y)

def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future(model, current_value, steps=5):
    predictions = [current_value]
    for _ in range(steps):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]

# ---------- Flask Routes ----------
@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        city = request.form["city"]
        current_weather = get_current_weather(city)
        if not current_weather:
            return render_template("index.html", error="City not found or API error")

        historical_data = read_historical_data("weather.csv")
        X, y, le = prepare_data(historical_data)
        rain_model = train_rain_model(X, y)

        # Encode wind dir
        wind_deg = current_weather['wind_gust_dir'] % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75)
        ]
        compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
        compass_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

        current_df = pd.DataFrame([{
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': compass_encoded,
            'WindGustSpeed': current_weather['WindGustSpeed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['current_temp'],
        }])
        rain_prediction = rain_model.predict(current_df)[0]

        # Future Predictions
        X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
        X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
        temp_model = train_regression_model(X_temp, y_temp)
        hum_model = train_regression_model(X_hum, y_hum)

        future_temp = predict_future(temp_model, current_weather['current_temp'])
        future_humidity = predict_future(hum_model, current_weather['humidity'])

        # Future times
        tz = pytz.timezone("Asia/Kolkata")
        now = datetime.now(tz)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
        future_times = [(next_hour + timedelta(hours=i)).strftime("%H:%M") for i in range(5)]

        # Plot graphs
        if not os.path.exists("static"):
            os.mkdir("static")

        plt.figure()
        plt.plot(future_times, future_temp, marker="o")
        plt.title("Future Temperature Predictions")
        plt.xlabel("Time")
        plt.ylabel("Temperature (°C)")
        plt.savefig("static/temp.png")
        plt.close()

        plt.figure()
        plt.plot(future_times, future_humidity, marker="o", color="green")
        plt.title("Future Humidity Predictions")
        plt.xlabel("Time")
        plt.ylabel("Humidity (%)")
        plt.savefig("static/humidity.png")
        plt.close()

        return render_template("result.html", 
                               weather=current_weather, 
                               rain_prediction="Yes" if rain_prediction else "No",
                               future_times=future_times,
                               future_temp=future_temp,
                               future_humidity=future_humidity)

    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)
