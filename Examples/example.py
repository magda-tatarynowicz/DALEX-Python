# Keras sample model generation
#from keras_example import create_sample_model
#(data, labels, model, names) = create_sample_model()

# Random forest sample model generation
#from random_forest_example import create_random_forest_model
#(data, labels, model, names) = create_random_forest_model()

# Linear regression sample model generation
from linear_regression_example import create_sample_linear_regression
(data, labels, model, names) = create_sample_linear_regression()

# Dalex operations
from py_dalex import py_dalex
dalex = py_dalex()
print(data)
explainer = dalex.explain(model, data, labels, names)
model_performance = dalex.model_performance(explainer)
variable_importance = dalex.variable_importance(explainer)
prediction_breakdown = dalex.prediction_breakdown(explainer, data[0,], names)
variable_response = dalex.variable_response(explainer, names[0])

# Saving plots
from save_plot import save_plot
save_plot(model_performance, 'model_performance.png')
save_plot(variable_importance, 'variable_importance.png')
save_plot(prediction_breakdown, 'prediction_breakdown.png')
save_plot(variable_response, 'variable_response.png')