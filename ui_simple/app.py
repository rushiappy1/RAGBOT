from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

BACKEND_ASK_URL = "http://localhost:8000/ask"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "")
    try:
        resp = requests.post(BACKEND_ASK_URL, json={"question": question})
        resp.raise_for_status()
        data = resp.json()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)
