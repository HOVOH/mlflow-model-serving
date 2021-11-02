from flask import Flask, request, jsonify
from waitress import serve
import pandas as pd
from mlflowModelLoader import MLFlowModelLoader
import os

app = Flask(__name__)

modelsLabel = zip(os.environ["MLFLOW_PREDICTION_LABELS"].split(","), os.environ["MLFOW_PREDICTION_MODELS_URI"].split(","))
model = MLFlowModelLoader(os.environ["MLFLOW_TRACKING_URI"], modelsLabel)
model.load()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    tasks = data.get("tasks")
    if tasks:
        tasks = pd.DataFrame(data=tasks)
        predictions = model.predict(tasks, calc_uncertainty=request.args.get("uncertainty"))
        return jsonify({
            "annotations": predictions
        })
    else:
        return "no tasks"


@app.route("/reload", methods=["POST"])
def reload():
    model.load()
    return jsonify({"reload": "success"})


@app.route("/health", methods=["POST"])
def health():
    return jsonify({"status": "UP"})


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=9000)
