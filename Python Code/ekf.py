# paperbot ekf
addr = 'http://192.168.4.1/'

from math import sqrt
import math
from numpy.random import randn
import numpy as np

def h_helper(x, Lx, Ly):
    rx = x[0]
    ry = x[1]
    th = x[2]

    th_abs =abs(th)
    tht = math.atan2((Ly-ry),(Lx-rx))
    thb = math.atan2((ry),(Lx - rx))
    thr = math.atan2((Lx-rx),ry)
    thl = math.atan2(rx,ry)
    c = math.cos(th_abs)
    s = math.sin(th_abs)

    if (th >= 0 and th_abs <= tht and th_abs <= thr):
        y = [(Lx-rx)/c, ry/c, th]
    elif (th >= 0 and th_abs <= tht and th_abs >= thr):
        y = [(Lx-rx)/c, (Lx-rx)/s, th]
    elif (th >= 0 and th_abs >= tht and th_abs <= thr):
        y = [(Lx-ry)/s, ry/c, th]
    elif (th >= 0 and th_abs >= tht and th_abs >= thr):
        y = [(Lx-ry)/s, (Lx-rx)/s, th]
    elif (th <= 0 and th_abs <= thb and th_abs <= thl):
        y = [(Lx-rx)/c, ry/c, th]
    elif (th <= 0 and th_abs <= thb and th_abs >= thl):
        y = [(Lx-rx)/c, rx/s, th]
    elif (th <= 0 and th_abs >= thb and th_abs <= thl):
        y = [ry/s, ry/c, th]
    elif (th <= 0 and th_abs >= thb and th_abs >= thl):
        y = [ry/s, rx/s, th]

    return y

def hmatr(x, Lx, Ly):
    x0 = x
    y0 = h_helper(x0, Lx, Ly)

    rx = x[0]
    ry = x[1]
    th = x[2]

    th_abs =abs(th)
    rx0 = x0[0]
    ry0 = x0[1]
    th0 = x0[2]
    lx0 = y0[0]
    ly0 = y0[1]
    al0 = y0[2]
    c = math.cos(th0)
    s = math.sin(th0)
    th0_abs = abs(th0)
    s_abs = math.sin(th0_abs)

    tht = math.atan2((Ly-ry),(Lx - rx))
    thb = math.atan2((ry),(Lx - rx))
    thr = math.atan2((Lx-rx),ry)
    thl = math.atan2(rx,ry)

    if (th >= 0 and th_abs <= tht and th_abs <= thr):
        a = (Lx-rx0)*s/(c**2)
        b = ry0*s/(c**2)
        H = [[-1/c, 0, a],
        [0, 1/c, b],
        [0, 0, 1]]
        C = [[rx0/c - a*th0 + lx0],
        [-ry0/c - b*th0 + ly0],
        [0]]
    elif (th >= 0 and th_abs <= tht and th_abs >= thr):
        a = (Lx-rx0)*s/(c**2)
        b = (Lx-rx0)*c/(s_abs**2)
        H = [[-1/c, 0, a],
        [-1/s_abs, 0, -(th0/th0_abs)*b],
        [0, 0, 1]]
        C = [[rx0/c - a*th0 + lx0],
        [rx0/s_abs + b*th0_abs + ly0],
        [0]]
    elif (th >= 0 and th_abs >= tht and th_abs <= thr):
        a = (Ly-ry0)*c/(s_abs**2)
        b = ry0*s/(c**2)
        H = [[0, -1/s_abs, -(th0/th0_abs)*a],
        [0, 1/c, b],
        [0, 0, 1]]
        C = [[ry0/s_abs + a*th0_abs + lx0],
        [-ry0/c - b*th0 + ly0],
        [0]]
    elif (th >= 0 and th_abs >= tht and th_abs >= thr):
        a = (Ly-ry0)*c/(s_abs**2)
        b = (Lx-rx0)*c/(s_abs**2)
        H = [[0, -1/s_abs, (-th0/th0_abs)*a],
        [-1/s_abs, 0, (-th0/th0_abs)*b],
        [0, 0, 1]]
        C = [[ry0/s_abs + a*th0_abs + lx0],
        [rx0/s_abs + b*th0_abs + ly0],
        [0]]
    elif (th <= 0 and th_abs <= thb and th_abs <= thl):
        a = (Lx-rx0)*s/(c**2)
        b = ry0*s/(c**2)
        H = [[-1/c, 0, a],
        [0, 1/c, b],
        [ 0, 0, 1]]
        C = [[rx0/c - a*th0 + lx0],
        [-ry0/c - b*th0 + ly0],
        [0]]
    elif (th <= 0 and th_abs <= thb and th_abs >= thl):
        a = (Lx-rx0)*s/(c**2)
        b = rx0*c/(s_abs**2)
        H = [[-1/c, 0, a],
        [1/s_abs, 0, (-th/th_abs)*b],
        [0, 0, 1]]
        C = [[rx0/c - a*th0 + lx0],
        [-rx0/s_abs + b*th0_abs + ly0],
        [0]]
    elif (th <= 0 and th_abs >= thb and th_abs <= thl):
        a = ry0*c/(s_abs**2)
        b = ry0*s/(c**2)
        H = [[0, 1/s_abs, (-th0/th0_abs)*a],
        [0, 1/c, b],
        [0, 0, 1]]
        C = [[-ry0/s_abs + a*th0_abs + lx0],
        [-ry0/c - b*th0 + ly0],
        [0]]
    elif (th <= 0 and th_abs >= thb and th_abs >= thl):
        a = ry0*c/(s_abs**2)
        b = rx0*c/(s_abs**2)
        H = [[0, 1/s_abs, (-th0/th0_abs)*a],
        [1/s_abs, 0, (-th0/th0_abs)*b],
        [0, 0, 1]]
        C = [[-ry0/s_abs + a*th0_abs + lx0],
        [-rx0/s_abs + b*th0_abs + ly0],
        [0]]
    return H, C

def b_helper(x, u, Cv, Cr):
    W = np.sqrt((85/2)**2 + 115**2)
    offset = math.atan2(85/2,115)
    theta = x[2]
    tL = u[0]
    tR = u[1]
    p_str = (tL * tR) > 0
    c = math.cos(theta)
    s = math.sin(theta)
    a1 = theta - offset
    a2 = theta + Cr*tR - offset
    a = p_str*Cv*c*tR - (1-p_str)*W*(math.cos(a1) - math.cos(a2))
    d = p_str*Cv*s*tR + (1-p_str)*W*(math.sin(a2) - math.sin(a1))
    c = (1-p_str)*Cr*tR;
    b = np.array([a, d, c])
    return b

pi = 3.1415

Lx = 400
Ly = 300
Cv = 147
Cr = 90/0.5 * (pi/180)

x_init = np.array([20., 30., 5.])   # state initialization

R = 0.1 * np.identity(3)    #  measurement noise
Q = np.zeros((3,3))         # process noise

'''
Kalman Filtering
'''
P_pri = 0.01 * np.identity(3)
x_pri = x_init
x_pos = x_init
I = np.identity(3)

for i in range(1,11):
    tL = input("Enter tL: ")
    tR = input("Enter tR: ")
    u = np.array([int(tL)/1000, int(tR)/1000])
    lx = input("Enter lx: ")
    ly = input("Enter ly: ")
    y = np.array([float(lx), float(ly), x_pos[2]])

    # Kalman Gain Update
    H, C = hmatr(x_pos, Lx, Ly)
    H = np.array(H)
    C = np.array(C)
    k1 = P_pri.dot(H.T)
    k2 = np.linalg.inv((H.dot(P_pri).dot(H.T)) + R)
    K = k1.dot(k2)
    P_pos = (I - K.dot(H)).dot(P_pri)
    P_pri = P_pos + Q

    # Measurement and State Estimation Update
    b = b_helper(x_pri, u, Cv, Cr)
    x_pri = x_pos + b
    print(x_pri)
    H, C = hmatr(x_pri, Lx, Ly)
    H = np.array(H)
    C = np.array(C)
    x_pos = x_pri + (K.dot((y - H.dot(x_pri) - C.T).T)).T
    x_pos = x_pos.flatten()
    print(x_pos)
'''
    print(y)
    print(H.dot(x_pri))
    print(C.T)
    print(K)
    print(K.dot((y - H.dot(x_pri) - C.T).T))
'''
