import numpy as np

data = input().split(' ')
if data[6] == '3': ##parabola
    b = np.array([float(data[1]), float(data[3]), float(data[5])])
    a = np.array([[float(data[0])*float(data[0]), float(data[0]), float(1)],
                  [float(data[2])*float(data[2]), float(data[2]), float(1)],
                  [float(data[4])*float(data[4]), float(data[4]), float(1)]])
    x = np.linalg.solve(a,b)
    print('a,b,c =', x)
if data[6] == '2': ##hyberbola
    a2 = (float(data[1])**2 * float(data[2])**2 - float(data[0])**2 * float(data[3])**2) / float(data[1])**2 - float(data[3])**2
    b2 = (float(data[1])**2 * float(data[2])**2 - float(data[0])**2 * float(data[3])**2) / float(data[0])**2 - float(data[2])**2
    print('a2 =', a2, 'b2 =', b2)
if data[6] == '1': ##ellipse
    a2 = (float(data[0])**2 * float(data[3])**2 - float(data[2])**2 * float(data[1])**2) / float(data[3])**2 - float(data[1])**2
    b2 = (float(data[0])**2 * float(data[3])**2 - float(data[2])**2 * float(data[1])**2) / float(data[0])**2 - float(data[2])**2
    print('a2 =', a2, 'b2 =', b2)
    
