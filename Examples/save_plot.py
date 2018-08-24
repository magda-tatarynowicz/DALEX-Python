import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

def save_plot(data, filename):
    grdevices = importr('grDevices')
    grdevices.png(file=filename, width=512, height=512)
    graphics = importr('graphics')
    rprint = robjects.globalenv.get("print")
    performance_plot = graphics.plot(data)
    rprint(performance_plot)
    grdevices.dev_off()