import os
import numpy as np

values = np.linspace(13.0, 15.0, 11)
low_values = values[:-1]
high_values = values[1:]

for low, high in zip(low_values, high_values):
    os.system('sbatch submit_walker.sh {} {}'.format(low, high))
