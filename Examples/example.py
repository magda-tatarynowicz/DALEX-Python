# Keras sample model generation
from keras_example import create_sample_model
(data, labels, model, names) = create_sample_model()

# Dalex operations
from py_dalex import py_dalex
dalex = py_dalex()
explainer = dalex.explain(model, data, labels, names)
model_performance = dalex.model_performance(explainer)
variable_importance = dalex.variable_importance(explainer)
prediction_breakdown = dalex.prediction_breakdown(explainer, data[0,], names)

# Saving plots
from save_plot import save_plot
save_plot(model_performance, 'model_performance.png')
save_plot(variable_importance, 'variable_importance.png')
save_plot(prediction_breakdown, 'prediction_breakdown.png')