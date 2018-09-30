import numpy as np
import rpy2.robjects as robjects

class PredictFunctionGenerator:
    def generate_function(self, model):
        def predict_values(r_model, r_data):
            np_data = np.transpose(np.array(r_data))
            predicted = np.transpose(model.predict(np_data))
            if predicted.ndim == 2:
                # for Keras two dimensional array is obtained
                predicted = predicted[0]
            return robjects.vectors.FloatVector(predicted)
        return predict_values