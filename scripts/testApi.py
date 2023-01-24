#!/usr/bin/env python3
import numpy as np
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
from sympy import Point, Polygon, Line

# Main function
if __name__ == '__main__':
    shap = False
    ex = 2

    # RULE FOR POLYGON
    # Bottom_left, Bottom_right, Top-right, Top_left, Bottom_left

    if shap:
        if ex == 1:
            # EXAMPLE 1
            m = 1
            b = 0
            x1, x2 = 0, 10
            y1, y2 = m*x1 + b, m*x2 + b

            polygon = Polygon([(10, 10), (10, 20), (20, 20), (20, 10), (10, 10)])
            # polygon = Polygon([(0, 2), (1, 2), (1, 3), (0, 3), (0, 2)])
        else:
            # EXAMPLE 2
            m = 1.6666666269302368
            b = -958.6666870117188
            w = 1280
            h = 720
            x1, x2 = 0, w
            y1, y2 = m*x1 + b, m*x2 + b

            polygon = Polygon([(739, 459), (883, 459), (883, 665), (739, 665), (739, 459)])


        # EXECUTION
        line1 = LineString([(y1, x1), (y2, x2)])

        res = line1.intersects(polygon)
        print(str(res))

        # plt.rcParams["figure.figsize"] = [50.00, 50.00]
        # plt.rcParams["figure.autolayout"] = True

        x, y = polygon.exterior.xy
        plt.plot(x, y, c="red")
        plt.plot([x1, x2], [y1, y2], marker = 'o')
        plt.show()

    else:
        m = 1.6666666269302368
        b = -958.6666870117188
        w = 1280
        h = 720
        x1, x2 = 0, w
        y1, y2 = m*x1 + b, m*x2 + b

        # creating points using Point()
        p1, p2, p3, p4, p5 = map(Point, [(739, 459), (883, 459), (883, 665), (739, 665), (739, 459)])
        # p1, p2, p3, p4, p5 = map(Point, [(442, 678), (563, 678), (563, 451), (442, 451), (442, 678)])
        
        # creating polygon using Polygon()
        poly1 = Polygon(p1, p2, p3, p4, p5)
        
        line1 = Line(Point(x1, y1), Point(x2, y2))
        # using intersection()
        isIntersection = poly1.intersection(line1)
                                            
        print(isIntersection)



