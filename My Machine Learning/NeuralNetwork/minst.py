import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mping
import math

from neural_network import MultilayerPerceptron

data = pd.read_csv('')
numbers_to_display = 25
num_cells = math.ceil(math.sqrt(numbers_to_display))

plt.figure(figsize=(10, 10))
