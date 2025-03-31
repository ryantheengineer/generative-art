# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 12:22:26 2025

@author: Ryan.Larson
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random

class WhaleOptimizationArt:
    def __init__(self, width=800, height=800, num_whales=50, max_iterations=1000,
                 a_decrease_factor=0.5, spiral_constant=1.0, palette='ocean'):
        self.width = width
        self.height = height
        self.num_whales = num_whales
        self.max_iterations = max_iterations
        self.a_decrease_factor = a_decrease_factor  # Controls exploration/exploitation balance
        self.spiral_constant = spiral_constant      # Controls spiral shape
        
        # Create canvas to hold our artwork
        self.canvas = np.zeros((height, width, 3))
        
        # Initialize whale positions
        self.whales = []
        for _ in range(num_whales):
            self.whales.append({
                'position': np.array([random.uniform(0, width), random.uniform(0, height)]),
                'fitness': 0,
                'color': np.random.rand(3),  # Random color for each whale
                'history': []  # Track positions for drawing trails
            })
        
        # Create a target "prey" that whales will hunt
        # This will move during the optimization
        self.prey_position = np.array([width/2, height/2])
        
        # Create an interesting fitness landscape for evaluation
        self.fitness_landscape = self._create_fitness_landscape()
        
        # Set up color palette for visualization
        self.cmap = plt.cm.get_cmap(palette)
        
        # Current iteration counter
        self.current_iteration = 0
        
    def _create_fitness_landscape(self):
        """Create an interesting fitness landscape for whales to navigate"""
        x = np.linspace(0, 10, self.width)
        y = np.linspace(0, 10, self.height)
        X, Y = np.meshgrid(x, y)
        
        A = 10
        Z = 2 * A + (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))
        
        # # Create multiple wave patterns
        # Z = 10*np.sin(0.5*X) * np.cos(0.5*Y) + np.sin(0.1*X+2) * np.cos(0.3*Y) 
        # Z += np.sin(X+Y) + np.cos(X-Y)
        
        # # Add some circular "target" areas that whales might converge on
        # # for _ in range(3):
        # #     cx = random.uniform(1, 9)
        # #     cy = random.uniform(1, 9)
        # #     radius = random.uniform(0.5, 2.0)
        # #     Z += 2 * np.exp(-((X-cx)**2 + (Y-cy)**2) / radius**2)
        
        # # Normalize to [0, 1] range
        # Z = (Z - Z.min()) / (Z.max() - Z.min())
        
        return Z
    
    def _evaluate_fitness(self, x, y):
        """Evaluate the fitness at a position"""
        # Ensure coordinates are within bounds
        x_int = max(0, min(int(x), self.width - 1))
        y_int = max(0, min(int(y), self.height - 1))
        
        return self.fitness_landscape[y_int, x_int]
    
    def update(self):
        """Run one iteration of the whale optimization algorithm"""
        # Parameter a decreases linearly from 2 to 0 over iterations
        a = 2 - self.current_iteration * (2.0 / self.max_iterations) * self.a_decrease_factor
        
        # Update prey position (the best whale position)
        best_whale = max(self.whales, key=lambda w: w['fitness'])
        self.prey_position = best_whale['position'].copy()
        
        # Move each whale
        for whale in self.whales:
            # Store current position in history for drawing trails
            if len(whale['history']) > 20:  # Limit history length
                whale['history'].pop(0)
            whale['history'].append(whale['position'].copy())
            
            # Generate random parameters
            r = np.random.rand()
            A = 2 * a * np.random.rand() - a  # Coefficient for encircling prey
            C = 2 * np.random.rand()          # Coefficient for search
            l = np.random.uniform(-1, 1)      # Parameter for spiral movement
            p = np.random.rand()              # Probability for hunting strategy
            
            # Calculate distance to prey
            D = np.linalg.norm(C * self.prey_position - whale['position'])
            
            # Update position based on hunting strategy
            new_position = whale['position'].copy()
            
            if p < 0.5:
                # Encircling prey or searching
                if abs(A) < 1:
                    # Encircling prey
                    new_position = self.prey_position - A * D
                else:
                    # Searching (exploring)
                    # Select a random whale to follow
                    random_whale = random.choice(self.whales)
                    random_position = random_whale['position']
                    D_random = np.linalg.norm(C * random_position - whale['position'])
                    new_position = random_position - A * D_random
            else:
                # Bubble-net attack (exploitation) - spiral movement
                # This creates the characteristic spiral pattern inspired by whale hunting
                D_prime = np.linalg.norm(self.prey_position - whale['position'])
                spiral = D_prime * np.exp(self.spiral_constant * l) * np.cos(2 * np.pi * l)
                
                # Calculate angle to prey
                angle = np.arctan2(
                    self.prey_position[1] - whale['position'][1],
                    self.prey_position[0] - whale['position'][0]
                )
                
                # Spiral movement
                new_position[0] = self.prey_position[0] - spiral * np.cos(angle)
                new_position[1] = self.prey_position[1] - spiral * np.sin(angle)
            
            # Ensure new position is within bounds
            new_position[0] = max(0, min(new_position[0], self.width - 1))
            new_position[1] = max(0, min(new_position[1], self.height - 1))
            
            # Update whale position
            whale['position'] = new_position
            
            # Evaluate new fitness
            whale['fitness'] = self._evaluate_fitness(new_position[0], new_position[1])
            
            # Draw on canvas
            self._draw_whale_movement(whale)
        
        # Increment iteration counter
        self.current_iteration += 1
        
    def _draw_whale_movement(self, whale):
        """Draw the whale's movement on the canvas"""
        # Get current position
        x, y = whale['position']
        x_int, y_int = int(x), int(y)
        
        # Ensure position is within bounds
        if 0 <= x_int < self.width and 0 <= y_int < self.height:
            # Draw point at current position
            color_value = whale['fitness']
            color = self.cmap(color_value)[:3]
            
            # Add color to canvas with some blending
            self.canvas[y_int, x_int] = self.canvas[y_int, x_int] * 0.7 + np.array(color) * 0.3
            
            # Draw trail if we have history
            if len(whale['history']) > 1:
                prev_x, prev_y = whale['history'][-2]
                prev_x, prev_y = int(prev_x), int(prev_y)
                
                # Simple line drawing using Bresenham's algorithm
                line_points = self._bresenham_line(prev_x, prev_y, x_int, y_int)
                for px, py in line_points:
                    if 0 <= px < self.width and 0 <= py < self.height:
                        # Fade color based on distance from current position
                        fade = 0.1 + 0.2 * np.random.rand()  # Add some randomness
                        self.canvas[py, px] = self.canvas[py, px] * (1-fade) + np.array(color) * fade
    
    def _bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm for drawing lines between points"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
                
        return points
    
    def render(self, filename=None, show_whales=False, show_prey=False):
        """Render the current state of the canvas"""
        # Normalize the canvas for visualization
        normalized = np.copy(self.canvas)
        if normalized.max() > 0:
            normalized = normalized / normalized.max()
        
        plt.figure(figsize=(10, 10))
        plt.imshow(normalized)
        
        if show_whales:
            # Plot current whale positions
            whale_x = [w['position'][0] for w in self.whales]
            whale_y = [w['position'][1] for w in self.whales]
            plt.scatter(whale_x, whale_y, s=10, c='white', alpha=0.7)
        
        if show_prey:
            # Plot prey position
            plt.scatter([self.prey_position[0]], [self.prey_position[1]], 
                       s=100, c='yellow', marker='*', alpha=0.8)
        
        plt.axis('off')
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def run_simulation(self, iterations=None, save_interval=None):
        """Run the simulation for multiple iterations"""
        total_iterations = iterations if iterations else self.max_iterations
        
        for i in range(total_iterations):
            self.update()
            
            # Optionally save intermediate frames
            if save_interval and i % save_interval == 0:
                self.render(filename=f'whale_optimization_{i:04d}.png')
                
            if i % 10 == 0:
                print(f"Completed iteration {i}/{total_iterations}")
        
        # Final render
        return self.canvas

# Example usage
if __name__ == "__main__":
    # Create a whale optimization art generator
    whale_art = WhaleOptimizationArt(
        width=800,
        height=800,
        num_whales=30,
        max_iterations=500,
        a_decrease_factor=0.6,
        spiral_constant=1.2,
        palette='ocean'
    )
    
    # Run the simulation with periodic saving
    whale_art.run_simulation(iterations=500, save_interval=50)
    
    # Generate final image
    whale_art.render(
        filename='whale_optimization_final.png',
        show_whales=True,
        show_prey=True
    )
    print("Final image saved as 'whale_optimization_final.png'")