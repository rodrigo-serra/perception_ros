#!/usr/bin/env python3
import numpy as np
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt

# Main function
if __name__ == '__main__':
    m = 1
    b = 0
    x1, x2 = 0, 10
    y1, y2 = m*x1 + b, m*x2 + b

    line1 = LineString([(y1, x1), (y2, x2)])

    # polygon = Polygon([(x_top_left, y_top_left), (x_bottom_left, y_bottom_left), (x_top_right, y_top_right), (x_bottom_right, y_bottom_right)])
    polygon = Polygon([(10, 10), (10, 20), (20, 20), (20, 10), (10, 10)])
    # polygon = Polygon([(0, 2), (1, 2), (1, 3), (0, 3), (0, 2)])

    
    
    # polygon = Polygon([(739, 459), (883, 459), (883, 665), (739, 665), (739, 459)])

    res = line1.intersects(polygon)
    print(str(res))

    # plt.rcParams["figure.figsize"] = [50.00, 50.00]
    # plt.rcParams["figure.autolayout"] = True

    x, y = polygon.exterior.xy
    plt.plot(x, y, c="red")
    plt.plot([x1, x2], [y1, y2], marker = 'o')
    plt.show()
