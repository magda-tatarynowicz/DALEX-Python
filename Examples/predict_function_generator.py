import numpy as np
import rpy2.robjects as robjects

class predict_function_generator:
    def generate_function(self, model):
        def predict_values(r_model, r_data):
            np_data = np.array(r_data)
            predicted = model.predict(np_data)
            return robjects.vectors.FloatVector(predicted)
        return predict_values