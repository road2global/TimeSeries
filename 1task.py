import numpy as np

type = input('1-ellipse, 2-hyperbola, 3-parabola:\n')
x_1, y_1, x_2, y_2, x_3, y_3 = (float(i) for i in input('3 coordinates:\n').split())
if type == '3': ##parabola
    b = np.array([y_1, y_2, y_3])
    a = np.array([[x_1 * x_1, x_1, float(1)],
                  [x_2 * x_2, x_2, float(1)],
                  [x_3 * x_3, x_3, float(1)]])
    x = np.linalg.solve(a,b)
    print('a, b, c =', x)
if type == '2': ##hyberbola
    a2 = (y_1**2 * x_2**2 - x_1**2 * y_2**2) / y_1**2 - y_2**2
    b2 = (y_1**2 * x_2**2 - x_1**2 * y_2**2) / x_1**2 - x_2**2
    print('a2 =', a2, 'b2 =', b2)
if type == '1': ##ellipse
    a2 = (x_1**2 * y_2**2 - x_2**2 * y_1**2) / y_2**2 - y_1**2
    b2 = (x_1**2 * y_2**2 - x_2**2 * y_1**2) / x_1**2 - x_2**2
    print('a2 =', a2, 'b2 =', b2)
    
