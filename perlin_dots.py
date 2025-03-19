# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 13:42:44 2025
@author: Ryan.Larson
Modified to use Matplotlib instead of vpype
"""
import numpy as np
from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import random

# Parameters
width, height = 210, 210  # A4 paper in mm
margin = 10  # margin in mm
dots_x, dots_y = 210, 210  # number of dots
dot_size = 0.5  # size of dots in mm
noise_scale = 1  # scale of the noise (smaller = smoother)
noise_magnitude = 20  # how much the dots move due to noise

def generate_circle(center_x, center_y, radius, num_points=8):
    """Generate points for a small circle"""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    return np.column_stack((x, y))

def generate_perlin_dot_field():
    # Create base dot grid positions
    x_positions = np.linspace(margin, width - margin, dots_x)
    y_positions = np.linspace(margin, height - margin, dots_y)
    
    # Random seed for reproducibility
    seed = random.randint(0, 100)
    octaves = 4
    NoiseX = PerlinNoise(octaves=octaves, seed=seed)
    NoiseY = PerlinNoise(octaves=octaves, seed=seed + 100)
    
    # Generate dots with Perlin noise displacement
    circles = []
    
    for x in x_positions:
        for y in y_positions:
            # Get Perlin noise values for x and y displacement
            noise_x = NoiseX([x * noise_scale, y * noise_scale]) * noise_magnitude
            noise_y = NoiseY([x * noise_scale, y * noise_scale]) * noise_magnitude
            
            # Apply displacement
            final_x = x + noise_x
            final_y = y + noise_y
            
            # Add a small circle (dot) at the position
            circle_points = generate_circle(final_x, final_y, dot_size)
            circles.append(circle_points)
    
    return circles

def plot_dot_field(circles, show=True, save_path=None, figsize=(7, 7)):
    """Plot the dot field using matplotlib"""
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # Create a line collection for efficient plotting
    lc = LineCollection(circles, colors='black', linewidths=0.5)
    ax.add_collection(lc)
    
    ax.set_aspect('equal')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()
    
    return fig, ax

def save_to_svg(circles, filename):
    """Save the circles to an SVG file using matplotlib"""
    fig, ax = plt.subplots(figsize=(width/25.4, height/25.4))  # Convert mm to inches
    
    lc = LineCollection(circles, colors='black', linewidths=0.5)
    ax.add_collection(lc)
    
    ax.set_aspect('equal')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {filename}")

if __name__ == "__main__":
    # Generate the circles
    circles = generate_perlin_dot_field()
    
    # Preview
    plot_dot_field(circles, save_path="perlin_dot_field_preview.png")
    
    # Save to SVG for plotting
    save_to_svg(circles, "perlin_dot_field.svg")