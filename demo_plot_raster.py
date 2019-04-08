import numpy as np
import matplotlib.pyplot as plt
import utils as ut

# Relative path to data (change to where you saved them)
path = '../ANDA_Data_Challenge_2019/data/'

# Select the data
color = 'red'

data_block = ut.load_dataset(color, path=None)

ax = ut.plot_trial_raster(data_block)

plt.show()
