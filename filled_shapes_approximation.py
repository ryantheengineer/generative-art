# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 12:11:26 2025

@author: Ryan.Larson
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw
import random

# Load target image
TARGET_IMAGE_PATH = "your_target_image.jpg"
target_img = Image.open(TARGET_IMAGE_PATH).convert("RGB")
TARGET_SIZE = target_img.size
target_array = np.array(target_img)

# Parameters
POPULATION_SIZE = 50
NUM_GENERATIONS = 1000
NUM_SHAPES = 100  # Number of shapes per individual

# Define an individual as a set of randomly generated circles
class Individual:
    def __init__(self):
        self.shapes = [self.random_circle() for _ in range(NUM_SHAPES)]
        self.image = None
        self.fitness = None

    def random_circle(self):
        """Generate a random circle (x, y, radius, color, alpha)"""
        return (
            random.randint(0, TARGET_SIZE[0]),  # x
            random.randint(0, TARGET_SIZE[1]),  # y
            random.randint(5, 50),  # radius
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),  # color
            random.uniform(0.1, 1.0)  # alpha (transparency)
        )

    def render(self):
        """Draws the individual as an image"""
        img = Image.new("RGB", TARGET_SIZE, (255, 255, 255))
        draw = ImageDraw.Draw(img, "RGBA")

        for x, y, r, color, alpha in self.shapes:
            fill = (color[0], color[1], color[2], int(alpha * 255))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)

        self.image = np.array(img)
        return img

    def compute_fitness(self):
        """Calculates fitness as the inverse of the MSE to the target image"""
        if self.image is None:
            self.render()
        mse = np.mean((self.image - target_array) ** 2)
        self.fitness = 1 / (mse + 1e-10)  # Avoid division by zero

# Genetic Algorithm
def evolve(population):
    """Evolve the population using selection, crossover, and mutation"""
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    new_population = population[:10]  # Keep top 10
    while len(new_population) < POPULATION_SIZE:
        parent1, parent2 = random.sample(population[:20], 2)
        child = crossover(parent1, parent2)
        mutate(child)
        new_population.append(child)
    return new_population

def crossover(parent1, parent2):
    """Combine two individuals to create a child"""
    child = Individual()
    split = random.randint(0, NUM_SHAPES)
    child.shapes = parent1.shapes[:split] + parent2.shapes[split:]
    return child

def mutate(ind):
    """Randomly alter an individual"""
    for _ in range(5):  # Mutate 5 random shapes
        ind.shapes[random.randint(0, NUM_SHAPES - 1)] = ind.random_circle()

# Run Evolution
population = [Individual() for _ in range(POPULATION_SIZE)]
for generation in range(NUM_GENERATIONS):
    for individual in population:
        individual.render()
        individual.compute_fitness()
    
    best_individual = max(population, key=lambda ind: ind.fitness)
    print(f"Generation {generation}, Best Fitness: {best_individual.fitness:.6f}")
    
    if generation % 10 == 0:  # Save progress every 10 generations
        best_individual.render().save(f"output_{generation}.png")

    population = evolve(population)
