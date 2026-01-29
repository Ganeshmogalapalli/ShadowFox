from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("car_price_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    year = int(request.form['year'])
    present_price = float(request.form['present_price'])
    kms_driven = int(request.form['kms_driven'])
    fuel_type = int(request.form['fuel_type'])
    seller_type = int(request.form['seller_type'])
    transmission = int(request.form['transmission'])
    owner = int(request.form['owner'])

    # ⚠️ Feature order MUST match training
    input_data = np.array([[year, present_price, kms_driven,
                             fuel_type, seller_type,
                             transmission, owner]])

    prediction = model.predict(input_data)

    return render_template(
        "index.html",
        prediction_text=f"Estimated Car Price: ₹ {round(prediction[0], 2)} Lakhs"
    )

if __name__ == "__main__":
    app.run(debug=True)
