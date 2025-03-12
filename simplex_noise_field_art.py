# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 11:14:29 2025

@author: Ryan.Larson
"""

import numpy as np
import matplotlib.pyplot as plt
from opensimplex import OpenSimplex
from tqdm import tqdm

# Create a class to represent our particle system
class SimplexFlowField:
    def __init__(self, width, height, scale=0.005, particles=1000, steps=500, noise_dims=3):
        self.width = width
        self.height = height
        self.scale = scale  # Controls the "zoom level" of the noise
        self.num_particles = particles
        self.steps = steps
        self.noise_dims = noise_dims
        
        # Initialize simplex noise generators
        self.noise_x = OpenSimplex(seed=np.random.randint(0, 10000))
        self.noise_y = OpenSimplex(seed=np.random.randint(0, 10000))
        
        if noise_dims >= 3:
            self.noise_velocity = OpenSimplex(seed=np.random.randint(0, 10000))
        
        if noise_dims >= 4:
            self.noise_time = OpenSimplex(seed=np.random.randint(0, 10000))
        
        # Initialize particle positions randomly
        self.particles = np.random.rand(self.num_particles, 2)
        self.particles[:, 0] *= self.width
        self.particles[:, 1] *= self.height
        
        # Storage for particle trajectories
        self.trajectories = np.zeros((self.num_particles, self.steps, 2))
        self.trajectories[:, 0, :] = self.particles
    
    def sample_noise(self, x, y, z=0, t=0):
        """Sample the noise field at the given coordinates"""
        # Scale coordinates to get smoother noise
        x_scaled = x * self.scale
        y_scaled = y * self.scale
        z_scaled = z * self.scale
        t_scaled = t * self.scale
        
        # Get noise value for x direction (-1 to 1)
        if self.noise_dims >= 4:
            dx = self.noise_x.noise4(x_scaled, y_scaled, z_scaled, t_scaled)
        elif self.noise_dims >= 3:
            dx = self.noise_x.noise3(x_scaled, y_scaled, z_scaled)
        else:
            dx = self.noise_x.noise2(x_scaled, y_scaled)
        
        # Get noise value for y direction (-1 to 1)
        if self.noise_dims >= 4:
            dy = self.noise_y.noise4(x_scaled, y_scaled, z_scaled, t_scaled)
        elif self.noise_dims >= 3:
            dy = self.noise_y.noise3(x_scaled, y_scaled, z_scaled)
        else:
            dy = self.noise_y.noise2(x_scaled, y_scaled)
        
        return dx, dy
    
    def get_velocity(self, x, y, step):
        """Get velocity for a particle at the given position and time step"""
        if self.noise_dims >= 3:
            z = self.noise_velocity.noise2(x * self.scale * 0.5, y * self.scale * 0.5)
        else:
            z = 0
            
        if self.noise_dims >= 4:
            t = step * 0.01 * self.noise_time.noise2(x * self.scale * 0.3, y * self.scale * 0.3)
        else:
            t = step * 0.01
            
        # Get the flow direction from the noise field
        dx, dy = self.sample_noise(x, y, z, t)
        
        # Scale to control the speed
        velocity_factor = 2.0
        return dx * velocity_factor, dy * velocity_factor
    
    def simulate(self):
        """Simulate particle movement through the flow field"""
        for step in tqdm(range(1, self.steps)):
            for i in range(self.num_particles):
                x, y = self.trajectories[i, step-1]
                
                # Get velocity from the flow field
                dx, dy = self.get_velocity(x, y, step)
                
                # Update position
                new_x = x + dx
                new_y = y + dy
                
                # Optional: wrap around boundaries
                new_x = new_x % self.width
                new_y = new_y % self.height
                
                # Store new position
                self.trajectories[i, step, 0] = new_x
                self.trajectories[i, step, 1] = new_y
    
    def plot(self, filename=None, background_color='white', line_color='black', dpi=300):
        """Plot the particle trajectories"""
        plt.figure(figsize=(self.width/100, self.height/100), dpi=dpi)
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        
        # Set background color
        ax = plt.gca()
        ax.set_facecolor(background_color)
        plt.axis('off')
        
        # Plot each particle trajectory
        for i in range(self.num_particles):
            plt.plot(
                self.trajectories[i, :, 0], 
                self.trajectories[i, :, 1], 
                color=line_color, 
                linewidth=0.5, 
                alpha=0.8
            )
        
        plt.tight_layout(pad=0)
        
        if filename:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
            print(f"Saved to {filename}")
        
        plt.show()
    
    def export_svg(self, filename):
        """Export the particle trajectories as SVG for plotting"""
        plt.figure(figsize=(self.width/100, self.height/100), dpi=300)
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        
        ax = plt.gca()
        ax.set_facecolor('white')
        plt.axis('off')
        
        for i in range(self.num_particles):
            plt.plot(
                self.trajectories[i, :, 0], 
                self.trajectories[i, :, 1], 
                color='black', 
                linewidth=0.5, 
                alpha=0.8
            )
        
        plt.tight_layout(pad=0)
        plt.savefig(filename, format='svg', bbox_inches='tight', pad_inches=0)
        print(f"Exported SVG to {filename}")
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create a flow field with different configurations
    
    # Basic 2D flow field
    flow_field_2d = SimplexFlowField(
        width=800,
        height=800,
        scale=0.003,
        particles=200,
        steps=300,
        noise_dims=2
    )
    flow_field_2d.simulate()
    flow_field_2d.plot(filename="simplex_flow_2d.png")
    flow_field_2d.export_svg("simplex_flow_2d.svg")
    
    # 3D flow field with velocity variation
    flow_field_3d = SimplexFlowField(
        width=800,
        height=800,
        scale=0.004,
        particles=300,
        steps=400,
        noise_dims=3
    )
    flow_field_3d.simulate()
    flow_field_3d.plot(filename="simplex_flow_3d.png")
    flow_field_3d.export_svg("simplex_flow_3d.svg")
    
    # 4D flow field with time variation
    flow_field_4d = SimplexFlowField(
        width=800,
        height=800,
        scale=0.005,
        particles=400,
        steps=500,
        noise_dims=4
    )
    flow_field_4d.simulate()
    flow_field_4d.plot(filename="simplex_flow_4d.png")
    flow_field_4d.export_svg("simplex_flow_4d.svg")