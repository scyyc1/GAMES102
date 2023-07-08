import numpy as np
import random
import matplotlib.pyplot as plt

def v_line_parametric_2D(p,d):
    t = np.linspace(0, 1, 1000)
    x = p[0] + d[0] * t
    y = p[1] + d[1] * t
    plt.plot(x, y, 'b-')

    plt.xlabel('x')
    plt.ylabel('y')
    
def v_line_2D(p1, p2):
    d = p2 - p1
    v_line_parametric_2D(p1, d)

    plt.xlabel('x')
    plt.ylabel('y')
    return d

def v_triangle_2D(p1,p2,p3):
    v_line_2D(p1,p2)
    v_line_2D(p2,p3)
    v_line_2D(p1,p3)
    
def v_triangle_barycentre(p1,p2,p3):
    G = (p1 + p2 + p3) / 3
    plt.plot(G[0], G[1], 'o', color='r')
    return G

def linear_interpolation(a, b, t):
    return (1-t)*a + t*b

def distance_euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def triangle_area(A, B, C):
    return np.linalg.norm(np.cross(B-A, C-A)) / 2

def slope_2D(A, B):
    return (B[1]-A[1])/B[0]-A[0]