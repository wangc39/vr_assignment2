import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import Voronoi, voronoi_plot_2d


if __name__ == "__main__":

    x = [43, 20, 34, 18, 12, 32, 40, 4, 44, 30, 6, 47, 23, 13, 38, 48, 36, 46, 50, 37, 21, 7, 28, 25, 10]
    y = [3, 43, 47, 31, 30, 39, 9, 33, 49, 36, 21, 48, 14, 34, 41, 4, 1, 44, 18, 24, 20, 11, 27, 42, 13]

    points = [list(i) for i in zip(x, y)]
    vor = Voronoi(points=points)


    # plt.figure(figsize=(8, 8), facecolor='w')
    fig = voronoi_plot_2d(vor=vor)
    based_path = os.path.abspath(os.path.dirname(__file__)) # 获取代码运行的基本路径
    save_path="./figures/Voronoi2.png"
    plt.savefig(os.path.join(based_path, save_path))
    plt.show()