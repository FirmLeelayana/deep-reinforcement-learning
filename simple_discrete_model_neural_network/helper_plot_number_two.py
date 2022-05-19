import matplotlib
font = {'family' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt

x = [0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14]

y1 = [1.,-0.1, 1.11 ,-0.221, 1.2431, -0.36741 ,-0.595849  , -0.3445661 , -0.62097729, -0.31692498, -0.65138252, -0.28347923,-0.68817285 ,-0.24300986 ,-0.73268915]
y2 = [ 1. , 0.,  1.  ,0. , 1. , 0. ,-1.  ,0., -1.,  0. ,-1. , 0. ,-1.  ,0., -1.]
y3 = [ 1.  ,2. , 3.,  4. , 5.,  6. , 5. , 4. , 3. , 2.,  1.,  0. ,-1. ,-2. ,-3.]
y4 = [1. , 2.1, 3.31 ,4.641  ,6.1051, 7.71561, 7.487171 ,7.2358881  ,6.95947691 ,6.6554246 , 6.32096706, 5.95306377 , 5.54837014 ,5.10320716, 4.61352787]
y5 = [ 1.,-2.1  ,   1.31 ,  -2.441, 1.6851 , -2.85361, 0.138971 , 0.8471319  , 0.06815491  ,0.9250296,  -0.01753256  ,1.01928581, -0.1212144 , 1.13333584 ,-0.24666942]
y6 = [ 1., -2.,  1. ,-2. , 1. ,-2. , 3.,-2. ,-1.  ,2. ,-1.  ,2. ,-1., -2. , 3.]
y7 = [  1.  , 0. , -1. , -2. , -3.  ,-4. , -1.  , 2.  ,-1. , -4.  ,-7. ,-10. , -7. , -4. , -1.]
y8 = [ 1. ,  0.1 , -0.89 ,  -1.979 , -3.1769,-4.49459 , -1.944049 , 0.8615461 ,  3.94770071 , 1.34247078, -1.52328214 ,-0.67561035, 0.25682861 , 1.28251147 , 2.41076262]

plt.plot(x, y1, label = "a = -1.1, b = 1")
plt.plot(x, y2, label = "a = -1, b = 1")
plt.plot(x, y3, label = "a = 1, b = 1")
plt.plot(x, y4, label = "a = 1.1, b = 1")
plt.plot(x, y5, label = "a = -1.1, b = -1")
plt.plot(x, y6, label = "a = -1, b = -1")
plt.plot(x, y7, label = "a = 1, b = -1")
plt.plot(x, y8, label = "a = 1.1, b = -1")


plt.xlabel('Time steps')
plt.ylabel('trajectory')
plt.title('trajectory plots of a single test episode')
plt.legend(loc="upper left")
plt.show()