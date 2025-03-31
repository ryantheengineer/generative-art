import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
import perlin_noise

def generate_perlin_noise(width, height, scale=10):
    """Generate a Perlin noise grayscale image using the perlin_noise library."""
    noise = perlin_noise.PerlinNoise(octaves=4)
    noise_field = np.array([[noise([i / scale, j / scale]) for j in range(width)] for i in range(height)])
    noise_field = (noise_field - noise_field.min()) / (noise_field.max() - noise_field.min())
    return noise_field

def generate_halftone_layer(image, grid_spacing, max_circle_size, filter_size):
    """Generate the halftone effect by averaging pixel values over a local area."""
    height, width = image.shape
    smoothed_image = uniform_filter(image, size=filter_size)  # Smooth image to consider local intensity
    
    x_grid, y_grid = np.meshgrid(np.arange(0, width, grid_spacing), np.arange(0, height, grid_spacing))
    intensity = smoothed_image[y_grid, x_grid]
    
    # Map intensity to circle size
    circle_sizes = max_circle_size * (1 - intensity)
    # line_thickness = 1
    line_thickness = np.ones(circle_sizes.shape)
    # line_thickness = 1 + 2 * (1 - intensity)  # Thicker lines for darker regions
    
    return x_grid, y_grid, circle_sizes, line_thickness

def plot_halftone(x, y, sizes, thicknesses, ax):
    """Plot the halftone effect using circles."""
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            circle = plt.Circle((x[i, j], y[i, j]), sizes[i, j] / 2, fill=False, color='black', linewidth=thicknesses[i, j])
            ax.add_patch(circle)


width, height = 500, 500  # Canvas size
grid_spacing = 10  # Grid spacing for halftone
max_circle_size = grid_spacing * 0.8  # Max circle diameter
filter_size = grid_spacing  # Averaging size for local intensity

# Generate Perlin noise grayscale image
grayscale_image = generate_perlin_noise(width, height, scale=300)

# Generate halftone representation
x, y, sizes, thicknesses = generate_halftone_layer(grayscale_image, grid_spacing, max_circle_size, filter_size)

# Plot original grayscale Perlin noise
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(grayscale_image, cmap='gray', origin='upper')
axs[0].set_title("Grayscale Perlin Noise")
axs[0].axis("off")

# Plot halftone representation
axs[1].set_xlim(0, width)
axs[1].set_ylim(0, height)
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].set_frame_on(False)
axs[1].invert_yaxis()
axs[1].set_title("Halftone Representation")
plot_halftone(x, y, sizes, thicknesses, axs[1])

plt.show()
