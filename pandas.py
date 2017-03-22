#!/usr/bin/python


import numpy as np
import pandas as pd


data = pd.Series([0.25, 0.5, 0.75, 1.0])
data.values
data.index

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])

population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)

indA & indB  # intersection
indA | indB  # union
indA ^ indB  # symmetric difference
