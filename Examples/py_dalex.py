import rpy2.robjects as robjects
import rpy2.rinterface as rinterface
import rpy2.robjects.numpy2ri as numpy2ri
from rpy2.robjects.packages import importr
from predict_function_generator import predict_function_generator

class py_dalex:  
    def __init__(self):
        self.dalex = importr('DALEX')
        self.predict_function_generator = predict_function_generator()
        # For conversion between R and NumPy objects
        numpy2ri.activate()

    def explain(self, model, data, labels):
        predict_function = self.predict_function_generator.generate_function(model)
        r_model = robjects.ListVector({'label': 'Keras Sequential'})
        return self.dalex.explain(model=r_model, data=data, y=labels, predict_function=predict_function)

    def model_performance(self, explainer):
        return self.dalex.model_performance(explainer)

    def prediction_breakdown(self, explainer, observation):
        return self.dalex.prediction_breakdown(explainer, observation=observation)