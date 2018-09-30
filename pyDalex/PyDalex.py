import rpy2.robjects as robjects
import rpy2.rinterface as rinterface
import rpy2.robjects.numpy2ri as numpy2ri
from rpy2.robjects.packages import importr
from .PredictFunctionGenerator import PredictFunctionGenerator

class PyDalex:  
    def __init__(self):
        self.dalex = importr('DALEX')
        self.predict_function_generator = PredictFunctionGenerator()
        # For conversion between R and NumPy objects
        numpy2ri.activate()

    def explain(self, model, data, labels, names):
        predict_function = self.predict_function_generator.generate_function(model)
        r_model = robjects.ListVector({'label': 'Keras Sequential'})
        r_data = robjects.DataFrame({names[x]:  robjects.FloatVector(data[:,x]) for x in range(data.shape[1])})
        return self.dalex.explain(model=r_model, data=r_data, y=labels, predict_function=predict_function)

    def model_performance(self, explainer):
        return self.dalex.model_performance(explainer)

    def prediction_breakdown(self, explainer, observation, names):
        r_observation = robjects.DataFrame({names[x]:  robjects.FloatVector([observation[x]]) for x in range(observation.shape[0])})
        return self.dalex.prediction_breakdown(explainer, observation=r_observation)

    def variable_importance(self, explainer):
        return self.dalex.variable_importance(explainer)

    def variable_response(self, explainer, variable):
        return self.dalex.variable_response(explainer, variable)