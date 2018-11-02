#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import pandas as pd

fname = 'doadores.csv'
data = pd.read_csv(fname, na_values='_', encoding='utf-8')

values = data.values[:,4,]
dates  = data.values[:,5,]

N_TRAIN = len(values) 
y = values[0:N_TRAIN,]
x = dates[0:N_TRAIN:]

# Plot error over iterations
plt.figure()
#plt.scatter(x_test, x_train)
colors = np.random.rand(N_TRAIN)
area = (20 * np.random.rand(N_TRAIN))**2
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.ylabel('Amount')
plt.title('Number of stupid people')
plt.xlabel('Date')
plt.show()
