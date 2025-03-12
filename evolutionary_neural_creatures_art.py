import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import random
from tqdm import tqdm
import copy
import os
from datetime import datetime

class NeuralNetwork:
    def __init__(self, input_size=5, hidden_size=4, output_size=2, weights=None):
        """
        Simple neural network for creature brains
        
        Inputs: 
            - Current x, y position (normalized)
            - Distance to nearest resource
            - Angle to nearest resource
            - Current energy level
            
        Outputs:
            - Movement direction (angle)
            - Movement speed
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        if weights is None:
            # Initialize random weights
            self.weights_ih = np.random.uniform(-1, 1, (hidden_size, input_size))
            self.bias_h = np.random.uniform(-1, 1, (hidden_size, 1))
            self.weights_ho = np.random.uniform(-1, 1, (output_size, hidden_size))
            self.bias_o = np.random.uniform(-1, 1, (output_size, 1))
        else:
            # Use provided weights
            self.weights_ih = weights[0]
            self.bias_h = weights[1]
            self.weights_ho = weights[2]
            self.bias_o = weights[3]
    
    def forward(self, inputs):
        """Process inputs through the network to get outputs"""
        # Convert inputs to numpy array
        x = np.array(inputs).reshape(-1, 1)
        
        # Hidden layer
        hidden = np.dot(self.weights_ih, x) + self.bias_h
        hidden = np.tanh(hidden)  # tanh activation
        
        # Output layer
        output = np.dot(self.weights_ho, hidden) + self.bias_o
        output = np.tanh(output)  # tanh activation
        
        return output.flatten()
    
    def get_weights(self):
        """Return all weights of the network"""
        return [self.weights_ih, self.bias_h, self.weights_ho, self.bias_o]
    
    def mutate(self, mutation_rate=0.1, mutation_amount=0.2):
        """Mutate the weights of the network"""
        def mutate_array(arr):
            mask = np.random.random(arr.shape) < mutation_rate
            mutations = np.random.uniform(-mutation_amount, mutation_amount, arr.shape)
            arr[mask] += mutations[mask]
            return arr
        
        self.weights_ih = mutate_array(self.weights_ih)
        self.bias_h = mutate_array(self.bias_h)
        self.weights_ho = mutate_array(self.weights_ho)
        self.bias_o = mutate_array(self.bias_o)


class Resource:
    def __init__(self, x, y, energy=100, respawn=True, respawn_time=50):
        self.x = x
        self.y = y
        self.initial_energy = energy
        self.energy = energy
        self.respawn = respawn
        self.respawn_time = respawn_time  # New: time until respawn
        self.respawn_counter = 0  # New: counter for respawning
        self.is_depleted = False
    
    def consume(self, amount):
        """Creature consumes energy from this resource"""
        if self.is_depleted:
            return 0
        
        actual_amount = min(amount, self.energy)
        self.energy -= actual_amount
        
        if self.energy <= 0:
            self.is_depleted = True
        
        return actual_amount
    
    def update(self):
        """Update resource state - handle respawning"""
        if self.respawn and self.is_depleted:
            self.respawn_counter += 1
            if self.respawn_counter >= self.respawn_time:
                self.energy = self.initial_energy
                self.is_depleted = False
                self.respawn_counter = 0
                return True  # Resource respawned
        return False  # No change
    
    def reset(self):
        """Reset resource to initial state"""
        if self.respawn and self.is_depleted:
            self.energy = self.initial_energy
            self.is_depleted = False
            self.respawn_counter = 0


class Creature:
    def __init__(self, x, y, brain=None, color=None, max_age=400,  # Increased max_age 
                 energy=150, speed_cost=0.05, existence_cost=0.02):  # Reduced energy costs
        # Position
        self.x = x
        self.y = y
        self.prev_positions = [(x, y)]  # Store trajectory
        
        # Physical attributes
        self.max_speed = 5.0
        self.size = 2.0
        self.perception_radius = 300.0  # Increased perception
        self.energy = energy
        self.max_energy = energy * 2
        self.speed_cost = speed_cost  # Energy consumed per unit of speed (reduced)
        self.existence_cost = existence_cost  # Energy consumed per step just to exist (reduced)
        
        # Life cycle
        self.age = 0
        self.max_age = max_age
        self.is_alive = True
        self.resources_consumed = 0
        self.offspring_count = 0
        self.distance_traveled = 0
        
        # Neural network brain
        if brain is None:
            self.brain = NeuralNetwork()
        else:
            self.brain = brain
        
        # Visualization
        if color is None:
            self.color = [random.random(), random.random(), random.random()]
        else:
            self.color = color
        
        # Store a reduced trajectory for efficiency in long-lived creatures
        self.trajectory_sample_rate = 1  # Store every nth position
        self.trajectory_counter = 0
    
    def sense_environment(self, world_width, world_height, resources):
        """Gather sensory inputs for the neural network"""
        # Normalized position
        norm_x = self.x / world_width
        norm_y = self.y / world_height
        
        # Find nearest non-depleted resource
        nearest_dist = float('inf')
        nearest_angle = 0
        
        for resource in resources:
            if not resource.is_depleted:
                dx = resource.x - self.x
                dy = resource.y - self.y
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < nearest_dist:
                    nearest_dist = distance
                    nearest_angle = np.arctan2(dy, dx) / np.pi  # Normalize angle to [-1, 1]
        
        # Cap the distance perception and normalize
        if nearest_dist == float('inf'):
            nearest_dist = -1  # No resources visible
        else:
            nearest_dist = min(1.0, nearest_dist / self.perception_radius)
            
        # Normalized energy level
        norm_energy = self.energy / self.max_energy
        
        return [norm_x, norm_y, nearest_dist, nearest_angle, norm_energy]
    
    def update(self, world_width, world_height, resources):
        """Update the creature's state based on its brain and environment"""
        if not self.is_alive:
            return
        
        # Increase age
        self.age += 1
        
        # Die of old age
        if self.age >= self.max_age:
            self.is_alive = False
            return
        
        # Get sensory inputs
        inputs = self.sense_environment(world_width, world_height, resources)
        
        # Get brain's decision
        outputs = self.brain.forward(inputs)
        
        # Extract movement information
        angle = outputs[0] * np.pi  # Map from [-1, 1] to [-π, π]
        speed = (outputs[1] + 1) / 2 * self.max_speed  # Map from [-1, 1] to [0, max_speed]
        
        # Calculate movement
        dx = speed * np.cos(angle)
        dy = speed * np.sin(angle)
        
        # Update position with boundary checking
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Enforce world boundaries
        new_x = max(0, min(world_width, new_x))
        new_y = max(0, min(world_height, new_y))
        
        # Calculate actual distance moved
        actual_dx = new_x - self.x
        actual_dy = new_y - self.y
        actual_distance = np.sqrt(actual_dx**2 + actual_dy**2)
        
        # Update position
        self.x = new_x
        self.y = new_y
        
        # Store trajectory with sampling for efficiency
        self.trajectory_counter += 1
        if self.trajectory_counter >= self.trajectory_sample_rate:
            self.prev_positions.append((self.x, self.y))
            self.trajectory_counter = 0
        
        # Energy costs
        movement_cost = actual_distance * self.speed_cost
        self.energy -= (movement_cost + self.existence_cost)
        self.distance_traveled += actual_distance
        
        # Check for resources
        self.consume_resources(resources)
        
        # Added: Minor passive energy regeneration
        if random.random() < 0.1:  # 10% chance of gaining a little energy
            self.energy += 0.1
            self.energy = min(self.energy, self.max_energy)
        
        # Check if dead from starvation
        if self.energy <= 0:
            self.is_alive = False
    
    def consume_resources(self, resources):
        """Consume energy from resources that the creature touches"""
        for resource in resources:
            if not resource.is_depleted:
                dx = resource.x - self.x
                dy = resource.y - self.y
                distance = np.sqrt(dx*dx + dy*dy)
                
                # If within consumption range (based on creature size)
                if distance < self.size * 2:
                    consumption_rate = 20.0  # Increased from 10.0
                    energy_gained = resource.consume(consumption_rate)
                    
                    self.energy += energy_gained
                    self.energy = min(self.energy, self.max_energy)  # Cap energy
                    
                    if energy_gained > 0:
                        self.resources_consumed += energy_gained
    
    def reproduce(self, mutation_rate=0.1):
        """Create an offspring with a potentially mutated brain"""
        # # Only reproduce if enough energy
        # if self.energy < self.max_energy * 0.4:  # Need at least 40% energy
        #     return None
        
        # Create a copy of the brain with mutations
        child_brain = NeuralNetwork(
            input_size=self.brain.input_size,
            hidden_size=self.brain.hidden_size,
            output_size=self.brain.output_size,
            weights=copy.deepcopy(self.brain.get_weights())
        )
        child_brain.mutate(mutation_rate)
        
        # Create color with small variation
        child_color = [
            min(1.0, max(0.0, self.color[0] + random.uniform(-0.1, 0.1))),
            min(1.0, max(0.0, self.color[1] + random.uniform(-0.1, 0.1))),
            min(1.0, max(0.0, self.color[2] + random.uniform(-0.1, 0.1)))
        ]
        
        # Create child with slightly varied attributes
        child = Creature(
            x=self.x + random.uniform(-10, 10),
            y=self.y + random.uniform(-10, 10),
            brain=child_brain,
            color=child_color,
            max_age=int(self.max_age * random.uniform(0.9, 1.1)),
            energy=self.energy * 0.3  # Child gets 30% of parent's energy
        )
        
        # Parent loses energy used to create offspring
        self.energy *= 0.7  # Parent keeps 70%
        self.offspring_count += 1
        
        return child
    
    def get_fitness(self):
        """Calculate the fitness score for this creature"""
        # Example fitness function - can be customized
        resource_factor = self.resources_consumed * 2
        distance_factor = self.distance_traveled * 0.1
        offspring_factor = self.offspring_count * 50
        survival_factor = self.age * 0.5  # Reward survival
        
        # Aesthetic factors - reward interesting paths
        turns = self.calculate_path_complexity()
        
        return resource_factor + distance_factor + offspring_factor + turns * 5 + survival_factor
    
    def calculate_path_complexity(self):
        """Calculate how complex (interesting) the creature's path is"""
        if len(self.prev_positions) < 3:
            return 0
            
        turns = 0
        for i in range(2, len(self.prev_positions)):
            p1 = self.prev_positions[i-2]
            p2 = self.prev_positions[i-1]
            p3 = self.prev_positions[i]
            
            # Calculate vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calculate angle between vectors
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 * mag2 > 0.001:  # Avoid division by near-zero
                cos_angle = dot / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to prevent numerical errors
                angle = np.arccos(cos_angle)
                
                # Count significant turns
                if angle > 0.2:  # About 11 degrees
                    turns += 1
        
        return turns


class World:
    def __init__(self, width, height, num_creatures=50, num_resources=50):  # More resources
        self.width = width
        self.height = height
        self.creatures = []
        self.resources = []
        self.generation = 0
        self.generation_history = []
        self.step_counter = 0
        
        # Store trajectories across generations
        self.trajectory_history = []
        
        # Initialize creatures
        for _ in range(num_creatures):
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            self.creatures.append(Creature(x, y))
        
        # Initialize resources
        self.create_resources(num_resources)
    
    def create_resources(self, num_resources):
        """Create resources in the world"""
        for _ in range(num_resources):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            energy = random.uniform(100, 200)  # More energy in resources
            respawn_time = random.randint(20, 100)  # Random respawn time
            self.resources.append(Resource(x, y, energy, True, respawn_time))
    
    def reset_resources(self):
        """Reset all resources for a new generation"""
        for resource in self.resources:
            resource.reset()
    
    def update(self):
        """Update all creatures and resources in the world"""
        self.step_counter += 1
        
        # Update creatures
        for creature in self.creatures:
            creature.update(self.width, self.height, self.resources)
        
        # Update resources - handle respawning
        for resource in self.resources:
            resource.update()
        
        # Spontaneous reproduction for successful creatures
        if self.step_counter % 50 == 0:  # Every 50 steps
            for creature in self.creatures:
                if creature.is_alive and creature.energy > creature.max_energy * 0.7:
                    # Creature is thriving, chance to reproduce
                    if random.random() < 0.3:  # 30% chance
                        child = creature.reproduce()
                        if child:
                            self.creatures.append(child)
        
        # Add random new resources occasionally
        if self.step_counter % 100 == 0:  # Every 100 steps
            if len(self.resources) < 100:  # Cap resource count
                num_to_add = random.randint(1, 3)
                for _ in range(num_to_add):
                    x = random.uniform(0, self.width)
                    y = random.uniform(0, self.height)
                    energy = random.uniform(100, 200)
                    respawn_time = random.randint(20, 100)
                    self.resources.append(Resource(x, y, energy, True, respawn_time))
    
    def is_generation_complete(self):
        """Check if the current generation is complete"""
        alive_count = sum(1 for creature in self.creatures if creature.is_alive)
        return alive_count < max(1, len(self.creatures) * 0.1)  # Generation ends when < 10% alive
    
    def evolve_population(self, generation_size=None, elitism_percentage=0.2, reproduction_threshold=None):
        """Create a new generation through selection and reproduction"""
        print("Evolving population...")
        
        # IMPORTANT: Save the trajectories of the current generation before creating the new one
        self.save_generation_trajectories()
        
        if generation_size is None:
            generation_size = len(self.creatures)
        
        if reproduction_threshold is None:
            reproduction_threshold = max(5, len(self.creatures) // 4)
        
        # Calculate fitness for all creatures
        for creature in self.creatures:
            creature.get_fitness()
        
        # Store statistics about this generation
        self.record_generation_stats()
        
        # Sort creatures by fitness (descending)
        print("Sorting creatures by fitness...")
        sorted_creatures = sorted(self.creatures, key=lambda c: c.get_fitness(), reverse=True)
        
        # Determine how many elite creatures to keep
        print("Deciding how many elite creatures to keep...")
        num_elite = max(1, int(generation_size * elitism_percentage))
        
        # Select top creatures for reproduction
        print("Selecting top creatures for reproduction...")
        reproducing_creatures = sorted_creatures[:reproduction_threshold]
        
        # Create new population starting with elites
        new_population = []
        
        # Add elite creatures (direct copies with reset attributes)
        print("Producing new population with copies of elites...")
        for i in range(num_elite):
            if i < len(sorted_creatures):
                elite = copy.deepcopy(sorted_creatures[i])
                elite.prev_positions = [(elite.x, elite.y)]  # Reset trajectory
                elite.age = 0
                elite.is_alive = True
                elite.resources_consumed = 0
                elite.offspring_count = 0
                elite.distance_traveled = 0
                new_population.append(elite)
        
        # Fill the rest through reproduction
        print("Producing the rest of new population through reproduction...")
        while len(new_population) < generation_size:
            # Select parent based on fitness (tournament selection)
            tournament_size = min(3, len(reproducing_creatures))
            if tournament_size == 0:
                break
                
            candidates = random.sample(reproducing_creatures, tournament_size)
            parent = max(candidates, key=lambda c: c.get_fitness())
            
            # Create offspring with mutations
            child = parent.reproduce(mutation_rate=0.2)
            if child:
                new_population.append(child)
        
        # If we still don't have enough, add random creatures
        print("Adding random creatures if there aren't enough...")
        while len(new_population) < generation_size:
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            new_population.append(Creature(x, y))
        
        # Replace population
        self.creatures = new_population
        self.reset_resources()
        self.generation += 1
        self.step_counter = 0
    
    def save_generation_trajectories(self):
        """Save the trajectories of the current generation before evolving"""
        # Create a deep copy of creature data with relevant info for visualization
        generation_data = []
        for creature in self.creatures:
            if len(creature.prev_positions) >= 2:  # Only save if there's a trajectory
                creature_data = {
                    'positions': copy.deepcopy(creature.prev_positions),
                    'color': copy.deepcopy(creature.color),
                    'fitness': creature.get_fitness(),
                    'alive': creature.is_alive
                }
                generation_data.append(creature_data)
        
        # Add to history
        self.trajectory_history.append({
            'generation': self.generation,
            'creatures': generation_data
        })
    
    def record_generation_stats(self):
        """Record statistics for the current generation"""
        print("Recording generation stats...")
        if not self.creatures:
            return
            
        fitness_values = [c.get_fitness() for c in self.creatures]
        distances = [c.distance_traveled for c in self.creatures]
        resources = [c.resources_consumed for c in self.creatures]
        alive_count = sum(1 for c in self.creatures if c.is_alive)
        
        stats = {
            'generation': self.generation,
            'avg_fitness': sum(fitness_values) / len(fitness_values),
            'max_fitness': max(fitness_values),
            'avg_distance': sum(distances) / len(distances),
            'avg_resources': sum(resources) / len(resources),
            'alive_ratio': alive_count / len(self.creatures)
        }
        
        self.generation_history.append(stats)
    
    def run_simulation(self, num_generations=10, steps_per_generation=300):
        """Run the simulation for multiple generations"""
        for gen in range(num_generations):
            print(f"Running Generation {self.generation + 1}/{num_generations}...")
            
            # Run steps for current generation
            for step in tqdm(range(steps_per_generation)):
                self.update()
                
                # If almost all creatures died, end early
                if self.is_generation_complete():
                    print(f"Generation ended after {step} steps with few creatures alive")
                    break
            
            # Create next generation
            self.evolve_population()
            
            # Report on generation
            if self.generation_history:
                stats = self.generation_history[-1]
                print(f"Gen {stats['generation']} - Avg Fitness: {stats['avg_fitness']:.2f}, "
                      f"Max Fitness: {stats['max_fitness']:.2f}, Survival: {stats['alive_ratio']*100:.1f}%")
    
    def visualize_generation(self, generation_idx=None, path_only=False, 
                             save_path=None, alpha=0.6, background_color='black',
                             show_points=True, line_width=1, point_size=1):
        """Visualize the paths of creatures from a specific generation"""
        print("Visualizing generation...")
        if generation_idx is None:
            generation_idx = self.generation - 1
        
        if generation_idx < 0 or len(self.trajectory_history) == 0:
            print(f"No data available for generation {generation_idx}")
            return
        
        # Find the correct generation data
        generation_data = None
        for gen_data in self.trajectory_history:
            if gen_data['generation'] == generation_idx:
                generation_data = gen_data
                break
        
        if generation_data is None:
            print(f"Could not find trajectory data for generation {generation_idx}")
            return
        
        plt.figure(figsize=(12, 12))
        ax = plt.gca()
        ax.set_facecolor(background_color)
        
        # Plot resources
        if not path_only:
            for resource in self.resources:
                circle = plt.Circle((resource.x, resource.y), 3, color='green', alpha=0.5)
                ax.add_patch(circle)
        
        # Sort creatures by fitness to draw more successful ones on top
        sorted_creatures = sorted(generation_data['creatures'], key=lambda c: c['fitness'])
        
        # Plot creature paths
        for creature_data in sorted_creatures:
            positions = creature_data['positions']
            if len(positions) < 2:
                continue
                
            # Create line segments
            points = np.array(positions)
            
            # Create line collection with creature's color
            if background_color == 'black':
                # Boost brightness for black backgrounds
                line_color = np.minimum(1.0, np.array(creature_data['color']) * 1.5)
                point_color = np.minimum(1.0, np.array(creature_data['color']) * 1.8)
            else:
                line_color = creature_data['color']
                point_color = creature_data['color']
            
            # Draw lines
            segments = np.concatenate([points[:-1, np.newaxis], points[1:, np.newaxis]], axis=1)
            lc = LineCollection(segments, colors=list(line_color).append(alpha), linewidths=line_width)
            # lc = LineCollection(segments, colors=[line_color + [alpha]], linewidths=line_width)
            ax.add_collection(lc)
            
            # Draw points if requested
            if show_points:
                plt.scatter(points[:, 0], points[:, 1], s=point_size, 
                           color=point_color, alpha=alpha)
        
        # Set plot limits
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.axis('off')
        plt.title(f"Generation {generation_idx} Trajectories", color='white' if background_color == 'black' else 'black')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def export_svg(self, filename, generation_idx=None, alpha=0.6, background_color='black',
                   show_points=True, line_width=0.5, point_size=0.5):
        """Export paths as SVG for plotting"""
        if generation_idx is None:
            generation_idx = self.generation - 1
        
        # Find the correct generation data
        generation_data = None
        for gen_data in self.trajectory_history:
            if gen_data['generation'] == generation_idx:
                generation_data = gen_data
                break
        
        if generation_data is None:
            print(f"Could not find trajectory data for generation {generation_idx}")
            return
        
        plt.figure(figsize=(self.width/100, self.height/100), dpi=300)
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        
        ax = plt.gca()
        ax.set_facecolor(background_color)
        plt.axis('off')
        
        # Sort creatures by fitness to draw more successful ones on top
        sorted_creatures = sorted(generation_data['creatures'], key=lambda c: c['fitness'])
        
        # Plot creature paths
        for creature_data in sorted_creatures:
            positions = creature_data['positions']
            if len(positions) < 2:
                continue
                
            # Create line segments
            points = np.array(positions)
            
            # Adjust color for visibility on background
            if background_color == 'black':
                # Boost brightness for black backgrounds
                line_color = np.minimum(1.0, np.array(creature_data['color']) * 1.5)
                point_color = np.minimum(1.0, np.array(creature_data['color']) * 1.8)
            else:
                line_color = creature_data['color']
                point_color = creature_data['color']
            
            # Draw lines
            segments = np.concatenate([points[:-1, np.newaxis], points[1:, np.newaxis]], axis=1)
            lc = LineCollection(segments, colors=[line_color + [alpha]], linewidths=line_width)
            ax.add_collection(lc)
            
            # Draw points if requested
            if show_points:
                plt.scatter(points[:, 0], points[:, 1], s=point_size, 
                           color=point_color, alpha=alpha)
        
        plt.tight_layout(pad=0)
        plt.savefig(filename, format='svg', bbox_inches='tight', pad_inches=0)
        print(f"Exported SVG to {filename}")
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create output directory
    output_dir = "neural_creature_art"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create world
    world = World(
        width=800, 
        height=800, 
        num_creatures=80,  # More creatures
        num_resources=200   # More resources
    )
    
    # Run simulation
    num_generations = 5
    steps_per_generation = 500  # More steps per generation
    world.run_simulation(num_generations, steps_per_generation)
    
    # Make sure to save the final generation's trajectories before visualization
    world.save_generation_trajectories()
    
    # Visualize final generation
    print("Visualizing final generation...")
    
    # Plot with lines only
    world.visualize_generation(
        save_path=f"{output_dir}/generation_{timestamp}_lines.png",
        background_color='black',
        path_only=True,
        show_points=False,
        alpha=0.7,
        line_width=1
    )
    
    # Plot with points only
    world.visualize_generation(
        save_path=f"{output_dir}/generation_{timestamp}_points.png",
        background_color='black',
        path_only=True,
        show_points=True,
        alpha=0.5,
        line_width=0,
        point_size=1
    )
    
    # Plot with both lines and points
    world.visualize_generation(
        save_path=f"{output_dir}/generation_{timestamp}_combined.png",
        background_color='black',
        path_only=True,
        show_points=True,
        alpha=0.6,
        line_width=0.5,
        point_size=1
    )
    
    # Export SVG
    world.export_svg(
        f"{output_dir}/generation_{timestamp}_final.svg",
        background_color='black',
        show_points=True
    )