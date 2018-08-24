# Keras sample model generation
from keras_example import create_sample_model
(data, labels, model) = create_sample_model()

# Dalex operations
from py_dalex import py_dalex
dalex = py_dalex()
explainer = dalex.explain(model, data, labels)
model_performance = dalex.model_performance(explainer)

# Saving plots
from save_plot import save_plot
save_plot(model_performance, 'model_performance.png')