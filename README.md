# ✈️ Bangladesh International Flight Fare Predictor

---

## 📄 Project Report (Google Sheets)
[![View Report](https://img.shields.io/badge/Google%20Sheets-View%20Report-brightgreen?style=for-the-badge&logo=google-sheets)](https://drive.google.com/drive/u/1/folders/1aHUnZQ5LcVOl5-S9AzuoOa51APKj6vJR)


Machine Learning–powered web application to analyze and predict international flight fares from Bangladesh.
Built using **Python, Streamlit, SQLite, Joblib, Scikit-Learn**.

---
## 🚀 Live App (Still Updating...)

👉 [Launch the Airline Price Predictor WebApp](https://balram-airline-flight-fare-predictor.streamlit.app/)

---

## 🚀 Project Overview

This project aims to predict the **Total Flight Fare (BDT)** for international routes originating from Bangladesh.
The system reads stored flight data, performs analytics, visualizations, and uses a **Random Forest Regressor** model to estimate ticket prices.

The deployed Streamlit application provides:

* 🔍 Data exploration & filtering
* 📊 Interactive charts
* 🤖 ML-based fare prediction
* 🗃️ SQLite database integration
* 📥 Download filtered data

---

## 🛠️ Technologies Used

* **Python 3.10+**
* **Streamlit** – UI
* **SQLite** – database
* **Scikit-Learn** – machine learning
* **Pandas / NumPy** – data handling
* **Plotly / Matplotlib / Seaborn** – charts
* **Joblib** – model loading

---

## 📁 Project Structure

```
📦 Flight-Fare-Prediction/
│
├── webapp.py                 # Streamlit app
├── flight_fare.db            # SQLite database (flight_data table)
├── Airline_rf_model.joblib   # Trained ML model
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

---

## 🧠 Machine Learning Model

Model Type: **Random Forest Regressor**
Target Variable:

* `Total Fare (BDT)`

### **Training Features**

```
Base Fare (BDT)
Tax & Surcharge (BDT)
Aircraft_Label
flight_mins
Class_Label
holiday_flag
Seasonality_Label
Destination_Label
Departure_Minute
Is_Night_Flight
Airline_Label
Booking_Label
Source_Label
Departure_Hour
Is_Premium_Airline
Departure_Month
```

Model saved as:

```
Airline_rf_model.joblib
```

---

## 🗃️ Database

SQLite DB: **flight_fare.db**

Required table:

```
flight_data
```

Must contain columns such as:

* Airline, Source, Destination
* Base Fare (BDT), Tax & Surcharge (BDT), Total Fare (BDT)
* All encoded label columns (e.g., Class_Label)

---

## 👨‍💻 Author

**Balram Shah**

**Linkedin : https://www.linkedin.com/in/balram-shah/**

Flight Fare Prediction & Data Analysis Project

2025

---

## ⭐ Show Your Support

If you like this project, don’t forget to **star ⭐ the repository**!

Happy Coding! ✨
