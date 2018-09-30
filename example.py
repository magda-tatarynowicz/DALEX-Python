# Keras sample model generation
#from sample_models import KerasExample
#(data, labels, model, names) = KerasExample().create_sample_model()

# Random forest sample model generation
#from sample_models import RandomForestExample
#(data, labels, model, names) = RandomForestExample().create_random_forest_model()

# Linear regression sample model generation
from sample_models import LinearRegressionExample
(data, labels, model, names) = LinearRegressionExample().create_sample_linear_regression()

# Dalex operations
from pyDalex import PyDalex
dalex = PyDalex()

explainer = dalex.explain(model, data, labels, names)
model_performance = dalex.model_performance(explainer)
variable_importance = dalex.variable_importance(explainer)
prediction_breakdown = dalex.prediction_breakdown(explainer, data[0,], names)
variable_response = dalex.variable_response(explainer, names[0])

# Saving plots
from save_plot import save_plot
save_plot(model_performance, 'plots/model_performance.png')
save_plot(variable_importance, 'plots/variable_importance.png')
save_plot(prediction_breakdown, 'plots/prediction_breakdown.png')
save_plot(variable_response, 'plots/variable_response.png')