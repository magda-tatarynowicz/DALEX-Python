# Keras model generation
from keras_example import create_sample_model
(data, labels, model) = create_sample_model()

# Import necessary modules
import rpy2.robjects as robjects
import rpy2.rinterface as rinterface
from rpy2.robjects.packages import importr
import numpy as np

# For conversion between R and NumPy objects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

# Function to predict probability passed to DALEX
@rinterface.rternalize
def predict_values(r_model, r_data):
    np_data = np.array(r_data)
    predicted = model.predict(np_data)
    return robjects.vectors.FloatVector(predicted)

# DALEX explainer and model performance created
dalex = importr('DALEX')
r_model = robjects.ListVector({'label': 'Keras Sequential'})
explainer = dalex.explain(model=r_model, data=data, y=labels, predict_function=predict_values)
model_performance = dalex.model_performance(explainer)

# Saving model_performance plot to file
grdevices = importr('grDevices')
grdevices.png(file='model_performance.png', width=512, height=512)
graphics = importr('graphics')
rprint = robjects.globalenv.get("print")
performance_plot = graphics.plot(model_performance)
rprint(performance_plot)
grdevices.dev_off()