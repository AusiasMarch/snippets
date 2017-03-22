#!/usr/bin/python

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

fig = plt.figure()
ax = plt.axes()

fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x))

#or with pylab
plt.plot(x, np.sin(x))
#If we want to create a single figure with multiple lines, we can simply call the plot function multiple times
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))



#colors
plt.plot(x, np.sin(x - 0), color='blue')        # specify color by name
plt.plot(x, np.sin(x - 1), color='g')           # short color code (rgbcmyk)
plt.plot(x, np.sin(x - 2), color='0.75')        # Grayscale between 0 and 1
plt.plot(x, np.sin(x - 3), color='#FFDD44')     # Hex code (RRGGBB from 00 to FF)
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3)) # RGB tuple, values 0 to 1
plt.plot(x, np.sin(x - 5), color='chartreuse'); # all HTML color names supported


#style
plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted');

# For short, you can use the following codes:
plt.plot(x, x + 4, linestyle='-')  # solid
plt.plot(x, x + 5, linestyle='--') # dashed
plt.plot(x, x + 6, linestyle='-.') # dashdot
plt.plot(x, x + 7, linestyle=':');  # dotted


#color and style
plt.plot(x, x + 0, '-g')  # solid green
plt.plot(x, x + 1, '--c') # dashed cyan
plt.plot(x, x + 2, '-.k') # dashdot black
plt.plot(x, x + 3, ':r');  # dotted red


#Adjusting the Plot
#Axes Limits
plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)

#axis to be displayed in reverse
plt.xlim(10, 0)
plt.ylim(1.2, -1.2)

#axis
plt.axis([-1, 11, -1.5, 1.5])#axis limits one call, ojo no axes axIs
plt.axis('tight')#tighten the bounds around the current plot
plt.axis('equal')#ensuring an equal aspect ratio
plt.grid(b=True)

#Labels
plt.title("A Sine Curve")
plt.xlabel("x")
plt.ylabel("sin(x)")

#legend
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')

plt.legend()

#in general de plt. a ax. es directo, plt.plot() → ax.plot()
#pero
#plt.xlabel() → ax.set_xlabel()
#plt.ylabel() → ax.set_ylabel()
#plt.xlim() → ax.set_xlim()
#plt.ylim() → ax.set_ylim()
#plt.title() → ax.set_title()
ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set(xlim=(0, 10), ylim=(-2, 2),#con ax.set(...) se puede poner todo de golpe
       xlabel='x', ylabel='sin(x)',
       title='A Simple Plot')
