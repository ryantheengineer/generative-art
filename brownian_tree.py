# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 11:32:12 2025

@author: Ryan.Larson
"""

import random
import matplotlib.pyplot as plt

def random_step():
    return random.choice([-1, 1])

def is_stuck(grid, x, y):
  
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if (dx != 0 or dy != 0) and 0 <= x + dx < len(grid) and 0 <= y + dy < len(grid[0]) and grid[x + dx][y + dy] == 1:
                return True
    return False

def generate_brownian_tree(size, num_particles):
    grid = [[0] * size for _ in range(size)]
    center = size // 2
    grid[center][center] = 1
    
    for _ in range(num_particles):
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        
        while 0 <= x < size and 0 <= y < size and not is_stuck(grid, x, y):
            x += random_step()
            y += random_step()
            
        if 0 <= x < size and 0 <= y < size:
          grid[x][y] = 1
          
    return grid

def visualize_tree(grid):
    plt.imshow(grid, cmap='gray_r', interpolation='none')
    plt.title('Brownian Tree')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    size = 200
    num_particles = 3000
    tree = generate_brownian_tree(size, num_particles)
    visualize_tree(tree)