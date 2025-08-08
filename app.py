from flask import Flask, render_template
import pickle

app = Flask(__name__)

@app.route("/")
def index():
    with open("pca_ckd.pkl", "rb") as f:
        scaler, pca, components = pickle.load(f)
    data = [{"pc1": round(row[0], 2), "pc2": round(row[1], 2)} for row in components[:10]]
    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)
