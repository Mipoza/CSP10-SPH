import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

def plot_1d():
    pos = pd.read_csv("data_1d/pos.csv").values.T[0]
    neighbors = pd.read_csv("data_1d/neighbors.csv").values.T[0]
    particle = pd.read_csv("data_1d/particle.csv").values.T[0]
    r = pd.read_csv("data_1d/rad.csv").values[0][0]

    fig, ax = plt.subplots()
    
    # ax.scatter(pos, np.zeros_like(pos), 
    #            c = 'black', s = 4, alpha = 0.6, marker = '|')
    ax.vlines(x = pos, ymin = 0, ymax = 1, color = 'black', alpha = 0.2)
    ax.vlines(x = neighbors, ymin = 0, ymax = 1, color = 'blue')
    ax.vlines(x = particle, ymin = 0, ymax = 1, color = 'red')
    
    ax.vlines(x = (particle[0] - r), ymin = 0, ymax = 1, color = 'violet') 
    ax.vlines(x = (particle[0] + r), ymin = 0, ymax = 1, color = 'violet')
    
    ax.get_yaxis().set_visible(False)
    
    fig.tight_layout()
    plt.savefig("nn_1d.pdf")
    plt.clf()

def plot_2d():
    pos = pd.read_csv("data_2d/pos.csv").values.T
    neighbors = pd.read_csv("data_2d/neighbors.csv").values.T
    particle = pd.read_csv("data_2d/particle.csv").values.T
    r = pd.read_csv("data_2d/rad.csv").values[0]
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.scatter(pos[0], pos[1], s = 2, c = 'black', alpha = 0.1)
    ax.scatter(neighbors[0], neighbors[1], s = 5, c = 'blue')
    ax.scatter(particle[0], particle[1], s = 5, c = 'red')
    
    circ = plt.Circle((particle[0], particle[1]), r, 
                      color = 'red', fill = False)
    ax.add_patch(circ)
    
    fig.tight_layout()
    plt.savefig("nn_2d.pdf")
    plt.clf()
    
def plot_3d():
    pos = pd.read_csv("data_3d/pos.csv").values.T
    neighbors = pd.read_csv("data_3d/neighbors.csv").values.T
    particle = pd.read_csv("data_3d/particle.csv").values.T
    r = pd.read_csv("data_3d/rad.csv").values[0]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(pos[0], pos[1], pos[2], s = 2, color = 'black', alpha = 0.005)
    ax.scatter(neighbors[0], neighbors[1], neighbors[2], 
               s = 5, color = 'blue')
    ax.scatter(particle[0], particle[1], particle[2], 
               s = 5, c = 'red')
    
    # ax.add_patch(circ)
    # plt.show()

if __name__ == "__main__":
    plot_1d()
    plot_2d()
    # Does not give nice plots
    # plot_3d()
