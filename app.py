import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

# Initialize parameters
n_vehicles = 5    # Number of vehicles
n_pedestrians = 5  # Number of pedestrians
collision_threshold = 0.5  # Collision distance
max_steps = 200  # Number of steps in the simulation
collision_count = 0  # Counter for collisions

# Generate initial positions for vehicles and pedestrians
vehicles, _ = make_blobs(n_samples=n_vehicles, centers=1, cluster_std=0.5)
pedestrians, _ = make_blobs(n_samples=n_pedestrians, centers=1, cluster_std=0.3)

# Initialize velocities
vehicle_velocities = (np.random.rand(n_vehicles, 2) - 0.5) * 0.1
pedestrian_velocities = (np.random.rand(n_pedestrians, 2) - 0.5) * 0.3

# Initialize color arrays
vehicle_colors = np.array([[0, 0, 1, 1]] * n_vehicles)  # Blue for vehicles
pedestrian_colors = np.array([[0, 1, 0, 1]] * n_pedestrians)  # Green for pedestrians

# Function to update positions
def update_positions(vehicles, pedestrians):
    global vehicle_velocities, pedestrian_velocities
    
    # Update vehicle positions
    for i in range(n_vehicles):
        vehicles[i] += vehicle_velocities[i]
    
    # Update pedestrian positions
    for i in range(n_pedestrians):
        pedestrians[i] += pedestrian_velocities[i]
    
    # Reflect off walls
    vehicles = np.clip(vehicles, -3, 3)
    pedestrians = np.clip(pedestrians, -3, 3)
    
    return vehicles, pedestrians

# Function to detect collisions
def detect_collisions(vehicles, pedestrians):
    distances = cdist(vehicles, pedestrians, metric='euclidean')
    return np.where(distances < collision_threshold)

# Setup the animation
fig, ax = plt.subplots()
scat_vehicles = ax.scatter(vehicles[:, 0], vehicles[:, 1], c=vehicle_colors, marker='o', label="Vehicles")
scat_pedestrians = ax.scatter(pedestrians[:, 0], pedestrians[:, 1], c=pedestrian_colors, marker='s', label="Pedestrians")

# Set limits and labels
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("Collision Detection")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.grid()

# Collision count display
collision_text = ax.text(-2.5, 2.5, f'Collisions: {collision_count}', fontsize=12, color='black')

# Update function for animation
def animate(frame):
    global vehicles, pedestrians, collision_count, vehicle_colors, pedestrian_colors
    
    # Update positions
    vehicles, pedestrians = update_positions(vehicles, pedestrians)
    
    # Update scatter plot positions
    scat_vehicles.set_offsets(vehicles)
    scat_pedestrians.set_offsets(pedestrians)

    # Reset colors
    vehicle_colors[:, :] = [0, 0, 1, 1]  # Reset to blue
    pedestrian_colors[:, :] = [0, 1, 0, 1]  # Reset to green

    # Detect collisions
    collisions = detect_collisions(vehicles, pedestrians)

    # Update collision count and colors for colliding entities
    for vehicle_idx, pedestrian_idx in zip(collisions[0], collisions[1]):
        collision_count += 1  # Increment the collision counter
        # Change colors to indicate collision
        vehicle_colors[vehicle_idx] = [1, 0, 0, 1]  # Change vehicle color to red
        pedestrian_colors[pedestrian_idx] = [1, 1, 0, 1]  # Change pedestrian color to yellow

    # Update scatter colors
    scat_vehicles.set_facecolor(vehicle_colors)
    scat_pedestrians.set_facecolor(pedestrian_colors)

    # Update the collision count display
    collision_text.set_text(f'Collisions: {collision_count}')
    
    return scat_vehicles, scat_pedestrians, collision_text

# Run the animation
ani = FuncAnimation(fig, animate, frames=max_steps, interval=100, blit=False)

plt.legend()
plt.show()
