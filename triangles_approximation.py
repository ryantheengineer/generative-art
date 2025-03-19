# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 11:20:15 2025

@author: Ryan.Larson
"""

import numpy as np
import cv2
import random
from scipy.spatial import Delaunay
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# Load the target image
TARGET_IMAGE_PATH = "bridge.jpg"
target_img = cv2.imread(TARGET_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
height, width = target_img.shape  # OpenCV loads images as (height, width)

# Resize the image
scale_factor = 0.5
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)
target_img = cv2.resize(target_img, (new_width, new_height))

# Store dimensions properly
TARGET_SIZE = (new_width, new_height)  # Ensure consistency: (width, height)
target_array = np.array(target_img)  # Keep this in (height, width) format

# Parameters
POPULATION_SIZE = 100
NUM_GENERATIONS = 100
INITIAL_TRIANGLE_COUNT = 10000  # Approximate number of triangles
LINE_THICKNESS = 2  # Thickness of the triangle edges

def generate_initial_points(num_points):
    """Generate random edge-aligned and interior points based on image brightness"""
    h, w = target_array.shape
    points = set()
    points.update([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])

    while len(points) < num_points:
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        brightness = target_array[y, x] / 255.0
        probability = 1 - brightness
        if random.random() < probability:
            points.add((x, y))
    
    return np.array(list(points))

class Individual:
    def __init__(self, points=None):
        self.points = points if points is not None else generate_initial_points(INITIAL_TRIANGLE_COUNT)
        self.triangles = Delaunay(self.points).simplices
        self.image = None
        self.fitness = None
    
    def render(self):
        """Draws the individual as an image using OpenCV"""
        img = np.full((TARGET_SIZE[1], TARGET_SIZE[0]), 255, dtype=np.uint8)  # White background
    
        for tri in self.triangles:
            pts = np.array([self.points[i] for i in tri], np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=0, thickness=LINE_THICKNESS)
    
        self.image = img
        return img


    # def compute_fitness(self):
    #     """Calculates fitness as the inverse of the Mean Squared Error (MSE)"""
    #     if self.image is None:
    #         self.render()
    #     mse = np.mean((self.image - target_array) ** 2)
    #     self.fitness = 1 / (mse + 1e-10)
        
    def compute_fitness(self):
        """Calculates fitness using Structural Similarity Index (SSIM)"""
        if self.image is None:
            self.render()
        
        # Compute SSIM (higher is better)
        ssim_score = ssim(self.image, target_array, data_range=target_array.max() - target_array.min())
        
        # Set fitness directly to SSIM score
        self.fitness = ssim_score

def evolve(population, progress_bar):
    """Evolve the population using selection, crossover, and mutation"""
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    new_population = population[:5]
    while len(new_population) < POPULATION_SIZE:
        
        end_index = int(len(population) / 2)
        parent1, parent2 = random.sample(population[:end_index], 2)
        child = crossover(parent1, parent2)
        mutate(child)
        new_population.append(child)
        progress_bar.update(1)
    return new_population

def crossover(parent1, parent2):
    """Combine two individuals to create a child"""
    split = random.randint(0, len(parent1.points) - 1)
    child_points = np.vstack((parent1.points[:split], parent2.points[split:]))
    return Individual(child_points)

def mutate(ind):
    """Randomly alter an individual"""
    for _ in range(random.randint(1, 5)):
        i = random.randint(0, len(ind.points) - 1)
        x, y = ind.points[i]
        x = np.clip(x + random.randint(-10, 10), 0, TARGET_SIZE[0] - 1)
        y = np.clip(y + random.randint(-10, 10), 0, TARGET_SIZE[1] - 1)
        ind.points[i] = (x, y)
    
    if random.random() < 0.1:
        if random.random() < 0.5 and len(ind.points) > 3:
            ind.points = np.delete(ind.points, random.randint(0, len(ind.points) - 1), axis=0)
        else:
            new_point = (random.randint(0, TARGET_SIZE[0] - 1), random.randint(0, TARGET_SIZE[1] - 1))
            ind.points = np.vstack((ind.points, new_point))
    
    ind.triangles = Delaunay(ind.points).simplices

population = [Individual() for _ in range(POPULATION_SIZE)]

for generation in range(NUM_GENERATIONS):
    print(f"\nGeneration {generation + 1}/{NUM_GENERATIONS}")
    with tqdm(total=POPULATION_SIZE, desc="Evaluating fitness", leave=False) as pbar:
        for individual in population:
            individual.render()
            individual.compute_fitness()
            pbar.update(1)
    best_individual = max(population, key=lambda ind: ind.fitness)
    print(f"  Best Fitness: {best_individual.fitness:.6f}")
    # if generation % 10 == 0:
    #     cv2.imwrite(f"triangles_output_{generation}.png", best_individual.image)
    cv2.imwrite(f"triangles_output_{generation}.png", best_individual.image)
    with tqdm(total=POPULATION_SIZE - 5, desc="Evolving population", leave=False) as pbar:
        population = evolve(population, pbar)
