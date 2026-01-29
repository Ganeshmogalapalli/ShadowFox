from flask import Flask, request, render_template
import pickle
import numpy as np
import webbrowser

app = Flask(__name__)

# Load trained Random Forest model
model = pickle.load(open("car_price_model.pkl", "rb"))

# Fixed year used during training (IMPORTANT for consistency)
CURRENT_YEAR = 2024

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    # --- Convert manufacturing year to car age ---
    manufacture_year = int(request.form['year'])
    years_used = CURRENT_YEAR - manufacture_year

    # --- Read other inputs ---
    present_price = float(request.form['present_price'])   # in Lakhs
    kms_driven = int(request.form['kms_driven'])

    fuel_type = int(request.form['fuel_type'])              # 0=Petrol, 1=Diesel, 2=CNG
    seller_type = int(request.form['seller_type'])          # 0=Dealer, 1=Individual
    transmission = int(request.form['transmission'])        # 0=Manual, 1=Automatic
    owner = int(request.form['owner'])

    # --- Feature order MUST match training ---
    input_data = np.array([[
        years_used,        # Year (car age)
        present_price,     # Present_Price
        kms_driven,        # Kms_Driven
        fuel_type,         # Fuel_Type
        seller_type,       # Seller_Type
        transmission,      # Transmission
        owner              # Owner
    ]])

    # --- Prediction ---
    prediction = model.predict(input_data)

    return render_template(
        "index.html",
        prediction_text=f"Estimated Car Price: ₹ {round(prediction[0], 2)} Lakhs"
    )

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)
