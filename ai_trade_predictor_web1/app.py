from flask import Flask, render_template, request
from model_logic import predict_trade

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        user_date = request.form["date"]
        prediction = predict_trade(user_date)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
#sala