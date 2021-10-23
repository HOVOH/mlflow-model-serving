import mlflow.pyfunc
from modAL.uncertainty import classifier_uncertainty
import numpy as np
import mlflow.sklearn


def prep_for_json(array):
    if isinstance(array, np.ndarray):
        array = array.tolist()
    elif not isinstance(array, list):
        array = list(array)
    return array


class MLFlowModelLoader:
    pipelines = []

    def __init__(self, tracking_uri, model_labels_tuples):
        mlflow.set_tracking_uri(tracking_uri)
        self.labels = model_labels_tuples

    def load(self):
        self.pipelines = []
        for label, model_uri in self.labels:
            print("Loading model %s (URI: %s)" % (label, model_uri))
            model = mlflow.sklearn.load_model(
                model_uri=model_uri
            )
            self.pipelines.append((label, model))

    def predict(self, data, calc_uncertainty=False):
        inferences = []
        for label, model in self.pipelines:
            predictions = model.predict(data)
            predictions = prep_for_json(predictions)
            inference = {
                "name": label,
                "predictions": predictions
            }
            if calc_uncertainty:
                uncertainty = classifier_uncertainty(model, data)
                uncertainty = prep_for_json(uncertainty)
                inference["uncertainty"] = uncertainty
            inferences.append(inference)
        return inferences

