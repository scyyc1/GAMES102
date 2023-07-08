import matplotlib.pyplot as plt
import numpy as np
import regression
import interpolation_gaussian
import interpolation_polynomial

fig, ax = plt.subplots()
ax.set_xlim([0, 10])  # Setting limit for better visualization
ax.set_ylim([0, 10])  # Setting limit for better visualization
points, = ax.plot([], [], color='red', marker='o', linestyle='')
polynomial, = ax.plot([], [], color='green', marker='o', markersize=0.1, linestyle='')
guassian, = ax.plot([], [], color='blue', marker='o', markersize=0.1, linestyle='')
reg, = ax.plot([], [], color='orange', marker='o', markersize=0.1, linestyle='')
r_reg, = ax.plot([], [], color='purple', marker='o', markersize=0.1, linestyle='')

is_stop = False

def onclick(event):
    x, y = event.xdata, event.ydata
    current_points = points.get_data()
    if not is_stop:
        new_points_x = list(current_points[0]) + [x]
        new_points_y = list(current_points[1]) + [y]
        points.set_data(new_points_x, new_points_y)

    fig.canvas.draw()

def on_keypress(event):
    if event.key == 't':
        print("Fit!")
        current_points = points.get_data()
        x_sample = np.array(current_points[0])
        y_sample = np.array(current_points[1])
        xmin = np.min(x_sample)
        xmax = np.max(x_sample)
        size = len(x_sample)
        x_range = np.linspace(xmin, xmax, 1000)

        poly = interpolation_polynomial.interpolation_polynomial(x_sample, y_sample)
        guass = interpolation_gaussian.interpolation_gaussian(x_sample, y_sample, 1)
        regress = regression.regression(x_sample, y_sample, 0, size-2)
        r_regress = regression.regression(x_sample, y_sample, 0.01, size-2)

        polynomial.set_data(x_range, poly.interpolate(x_range))
        guassian.set_data(x_range, guass.interpolate(x_range))
        reg.set_data(x_range, regress.interpolate(x_range))
        r_reg.set_data(x_range, r_regress.interpolate(x_range))


    elif event.key == 'c':
        print("Clear!")
        points.set_data([],[])
        polynomial.set_data([],[])
        guassian.set_data([],[])
        reg.set_data([],[])
        r_reg.set_data([],[])

    else:
        print(f"You pressed {event.key}.")
    
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', on_keypress)

plt.show()
