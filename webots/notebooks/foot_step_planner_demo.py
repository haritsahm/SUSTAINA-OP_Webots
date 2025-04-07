import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from IPython.display import HTML

# Check if matplotlib is available, otherwise just print information
try:
    # These imports are already at the top of the file, so we just check if they work
    if plt and Rectangle and FuncAnimation and animation and HTML:
        MATPLOTLIB_AVAILABLE = True
except NameError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Will only print information without plotting.")

# Try to import OmegaConf for configuration
try:
    from omegaconf import OmegaConf
except ImportError:
    print("OmegaConf not available. Will use default configuration.")
    OmegaConf = None
    # Create a simple configuration object
    class SimpleConfig:
        def __init__(self):
            self.walk = SimpleConfig()
            self.walk.t_step = 0.25
            self.walk.t_ds = 0.1
            self.walk.t_ss = 0.15
            self.walk.t_start = 0.25
            self.walk.t_end = 0.25
            self.walk.dt = 0.01
            self.walk.step_height = 0.02
            self.walk.foot_distance = 0.07

# Add the parent directory to the path so we can import the foot_step_planner module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'controllers', 'omnidirectional_walk', 'walk_engines')))

# Import the FootStepPlanner class
from foot_step_planner import FootStepPlanner, SupportLeg

# Load configuration
config_path = os.path.join(
    os.path.dirname(__file__), '..',
    'controllers', 'omnidirectional_walk', 'config.yaml'
)
config = OmegaConf.load(config_path)

print('Configuration loaded successfully')
print(config)

# Create the planner
planner = FootStepPlanner(config)

# Define different velocity commands to test
velocity_commands = [
    (0.5, 0.0, 0.0),  # Forward walking
    (0.0, 0.5, 0.0),  # Sideways walking
    (0.5, 0.5, 0.1),  # Diagonal walking with rotation
    (0.0, 0.0, 0.2),  # Turning in place
]

# Function to plot the footsteps using the new calculate_footsteps method output
def plot_footstep_sequence(step_sequence, plot_title, zmp_trajectory=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract left foot, right foot, and torso positions
    left_foot_positions = []
    right_foot_positions = []
    torso_positions = []
    next_left_foot_positions = []
    next_right_foot_positions = []
    next_torso_positions = []
    times = []
    
    for step in step_sequence:
        time = step['time']
        times.append(time)
        
        # Current positions
        left_foot = step['L']
        right_foot = step['R']
        torso = step['C']
        
        # Next positions
        next_left_foot = step['next_L']
        next_right_foot = step['next_R']
        next_torso = step['next_C']
        
        # Convert to flat arrays for easier plotting
        left_foot_positions.append(np.array(left_foot).ravel())
        right_foot_positions.append(np.array(right_foot).ravel())
        torso_positions.append(np.array(torso).ravel())
        next_left_foot_positions.append(np.array(next_left_foot).ravel())
        next_right_foot_positions.append(np.array(next_right_foot).ravel())
        next_torso_positions.append(np.array(next_torso).ravel())
    
    # Plot left foot trajectory
    left_x = [pos[0] for pos in left_foot_positions]
    left_y = [pos[1] for pos in left_foot_positions]
    ax.plot(left_x, left_y, 'b-', label='Left Foot Path')
    
    # Plot right foot trajectory
    right_x = [pos[0] for pos in right_foot_positions]
    right_y = [pos[1] for pos in right_foot_positions]
    ax.plot(right_x, right_y, 'r-', label='Right Foot Path')
    
    # Plot torso trajectory
    torso_x = [pos[0] for pos in torso_positions]
    torso_y = [pos[1] for pos in torso_positions]
    ax.plot(torso_x, torso_y, 'g-', label='Torso Path')
    
    # Plot ZMP trajectory if provided
    if zmp_trajectory is not None:
        zmp_x = [pos[1] for pos in zmp_trajectory]  # x-coordinate
        zmp_y = [pos[2] for pos in zmp_trajectory]  # y-coordinate
        ax.plot(zmp_x, zmp_y, 'k--', label='ZMP Trajectory', linewidth=1.5)
    
    # Draw foot rectangles at each step
    foot_length = 0.08  # Approximate foot length in meters
    foot_width = 0.04   # Approximate foot width in meters
    
    for i, (left_foot, right_foot, support_leg) in enumerate(
            zip(left_foot_positions, right_foot_positions, 
                [step['SF'] for step in step_sequence])):
        # Left foot rectangle
        left_rect = Rectangle(
            (left_foot[0] - foot_length/2, left_foot[1] - foot_width/2),
            foot_length, foot_width,
            angle=np.degrees(left_foot[2]),  # Convert radians to degrees
            color='blue', alpha=0.3 if support_leg != SupportLeg.LEFT else 0.7
        )
        ax.add_patch(left_rect)
        
        # Right foot rectangle
        right_rect = Rectangle(
            (right_foot[0] - foot_length/2, right_foot[1] - foot_width/2),
            foot_length, foot_width,
            angle=np.degrees(right_foot[2]),  # Convert radians to degrees
            color='red', alpha=0.3 if support_leg != SupportLeg.RIGHT else 0.7
        )
        ax.add_patch(right_rect)
        
        # Add step numbers and time labels with more details
        time_val = step_sequence[i]['time']
        support_text = "L" if support_leg == SupportLeg.LEFT else "R" \
            if support_leg == SupportLeg.RIGHT else "B"
        
        # Add more detailed text labels outside the foot rectangles
        ax.text(
            left_foot[0], left_foot[1] + 0.03, 
            f'L{i} (t={time_val:.2f}s)', 
            fontsize=8, ha='center', va='bottom', color='blue'
        )
        
        ax.text(
            right_foot[0], right_foot[1] - 0.03, 
            f'R{i} (t={time_val:.2f}s)', 
            fontsize=8, ha='center', va='top', color='red'
        )
        
        # Add support leg indicator
        if support_leg == SupportLeg.LEFT:
            ax.text(
                left_foot[0], left_foot[1], f'SUPPORT', 
                fontsize=7, ha='center', va='center', color='white', 
                bbox=dict(facecolor='blue', alpha=0.7, pad=1)
            )
        elif support_leg == SupportLeg.RIGHT:
            ax.text(
                right_foot[0], right_foot[1], f'SUPPORT', 
                fontsize=7, ha='center', va='center', color='white',
                bbox=dict(facecolor='red', alpha=0.7, pad=1)
            )
    
    # Set axis properties
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(plot_title)
    ax.grid(True)
    ax.legend()
    
    # Set axis limits with some padding
    all_x = left_x + right_x + torso_x
    all_y = left_y + right_y + torso_y
    x_min = min(all_x) - 0.1
    x_max = max(all_x) + 0.1
    y_min = min(all_y) - 0.1
    y_max = max(all_y) + 0.1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    return fig, ax

# Add a function to transform step sequence to global frame
def transform_step_sequence_to_global(step_sequence, global_frame):
    """
    Transform a step sequence from support foot frame to global frame.
    
    Parameters:
    -----------
    step_sequence : list of dict
        List of dictionaries containing footstep sequence data
    global_frame : numpy.ndarray
        Global frame reference position
    
    Returns:
    --------
    list of dict
        Transformed step sequence in global frame
    """
    global_step_sequence = []

    for step in step_sequence:
        # Create a new step dictionary with the same keys
        global_step = {}

        # Copy the time and support leg information
        global_step['time'] = step['time']
        global_step['SF'] = step['SF']

        # Transform the foot and torso positions to global frame
        global_step['L'] = planner.transform_to_frame(step['L'], global_frame)
        global_step['R'] = planner.transform_to_frame(step['R'], global_frame)
        global_step['C'] = planner.transform_to_frame(step['C'], global_frame)
        global_step['next_L'] = planner.transform_to_frame(step['next_L'], global_frame)
        global_step['next_R'] = planner.transform_to_frame(step['next_R'], global_frame)
        global_step['next_C'] = planner.transform_to_frame(step['next_C'], global_frame)

        global_step_sequence.append(global_step)

    return global_step_sequence

# Function to create an animation of the footstep sequence
def create_footstep_animation(step_sequence, filename, title, zmp_trajectory=None):
    """
    Create an animation of the footstep sequence and save it as an MP4 file.
    
    Parameters:
    -----------
    step_sequence : list of dict
        List of dictionaries containing footstep sequence data
    filename : str
        Filename to save the animation to
    title : str
        Title for the animation
    zmp_trajectory : list, optional
        List of [time, x, y] points defining the ZMP trajectory
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up foot rectangles
    foot_length = 0.08  # Approximate foot length in meters
    foot_width = 0.04   # Approximate foot width in meters
    
    # Extract time points for animation
    times = [step['time'] for step in step_sequence]
    max_time = times[-1]
    
    # Set axis properties
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True)
    
    # Create legend elements
    left_foot_patch = plt.Rectangle((0, 0), foot_length, foot_width, 
                                   color='blue', alpha=0.5, label='Left Foot')
    right_foot_patch = plt.Rectangle((0, 0), foot_length, foot_width, 
                                    color='red', alpha=0.5, label='Right Foot')
    torso_line, = ax.plot([], [], 'g-', label='Torso Path')
    left_line, = ax.plot([], [], 'b-', label='Left Foot Path')
    right_line, = ax.plot([], [], 'r-', label='Right Foot Path')
    
    # Add ZMP trajectory line if provided
    zmp_line = None
    if zmp_trajectory is not None:
        zmp_line, = ax.plot([], [], 'm--', label='ZMP Trajectory')
    
    # Create legend elements list
    legend_elements = [left_foot_patch, right_foot_patch, torso_line, left_line, right_line]
    if zmp_line is not None:
        legend_elements.append(zmp_line)
    
    # Add legend
    ax.legend(handles=legend_elements)
    
    # Calculate axis limits
    all_positions = []
    for step in step_sequence:
        all_positions.extend([step['L'][:2], step['R'][:2], step['C'][:2], 
                             step['next_L'][:2], step['next_R'][:2], step['next_C'][:2]])
    
    all_x = [pos[0] for pos in all_positions]
    all_y = [pos[1] for pos in all_positions]
    
    # Include ZMP trajectory in axis limits if provided
    if zmp_trajectory is not None:
        zmp_x = [pos[1] for pos in zmp_trajectory]  # x-coordinate
        zmp_y = [pos[2] for pos in zmp_trajectory]  # y-coordinate
        all_x.extend(zmp_x)
        all_y.extend(zmp_y)
    
    x_min, x_max = min(all_x) - 0.1, max(all_x) + 0.1
    y_min, y_max = min(all_y) - 0.1, max(all_y) + 0.1
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Initialize foot rectangles and text elements
    left_rect = plt.Rectangle((0, 0), foot_length, foot_width, 
                             color='blue', alpha=0.5)
    right_rect = plt.Rectangle((0, 0), foot_length, foot_width, 
                              color='red', alpha=0.5)
    
    left_text = ax.text(0, 0, '', fontsize=8, color='blue')
    right_text = ax.text(0, 0, '', fontsize=8, color='red')
    torso_text = ax.text(0, 0, '', fontsize=8, color='green')
    time_text = ax.text(x_min + 0.05, y_max - 0.05, '', fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.7))
    support_text = ax.text(x_min + 0.05, y_max - 0.10, '', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7))
    
    ax.add_patch(left_rect)
    ax.add_patch(right_rect)
    
    # Lists to store path data
    torso_path_x, torso_path_y = [], []
    left_path_x, left_path_y = [], []
    right_path_x, right_path_y = [], []
    
    # Initialize return elements list
    return_elements = [left_rect, right_rect, left_text, right_text, torso_text, time_text, support_text,
                      torso_line, left_line, right_line]
    
    # Add ZMP line to return elements if it exists
    if zmp_trajectory is not None:
        return_elements.append(zmp_line)
    
    def init():
        left_rect.set_xy((0, 0))
        right_rect.set_xy((0, 0))
        left_text.set_text('')
        right_text.set_text('')
        torso_text.set_text('')
        time_text.set_text('')
        support_text.set_text('')
        torso_line.set_data([], [])
        left_line.set_data([], [])
        right_line.set_data([], [])
        
        # Initialize ZMP line if it exists
        if zmp_trajectory is not None:
            zmp_line.set_data([], [])
            
        return tuple(return_elements)
    
    def animate(frame):
        # Calculate current time
        t = frame / 100 * max_time
        
        # Find the current step in the sequence
        current_step_idx = 0
        for idx, step_time in enumerate(times):
            if step_time <= t:
                current_step_idx = idx
        
        # Get current step data
        current_step = step_sequence[current_step_idx]
        
        # Initialize positions based on current step
        left_pos = np.array(current_step['L']).ravel()
        right_pos = np.array(current_step['R']).ravel()
        torso_pos = np.array(current_step['C']).ravel()
        
        # If we're at the last frame, use the final positions
        if current_step_idx == len(step_sequence) - 1:
            left_pos = np.array(current_step['next_L']).ravel()
            right_pos = np.array(current_step['next_R']).ravel()
            torso_pos = np.array(current_step['next_C']).ravel()
        
        # Update foot rectangles
        left_rect.set_xy((left_pos[0] - foot_length/2, left_pos[1] - foot_width/2))
        left_rect.angle = np.degrees(left_pos[2]) if len(left_pos) > 2 else 0
        
        right_rect.set_xy((right_pos[0] - foot_length/2, right_pos[1] - foot_width/2))
        right_rect.angle = np.degrees(right_pos[2]) if len(right_pos) > 2 else 0
        
        # Update text elements
        time_text.set_text(f'Time: {t:.2f}s')
        support_leg = current_step['SF']
        support_text.set_text(f'Support: {"LEFT" if support_leg == SupportLeg.LEFT else "RIGHT" if support_leg == SupportLeg.RIGHT else "BOTH"}')
        
        # Update path lines (show the path up to the current time)
        path_steps = [step for step in step_sequence if step['time'] <= t]
        
        if path_steps:
            # Update torso path
            torso_x = [np.array(step['C']).ravel()[0] for step in path_steps]
            torso_y = [np.array(step['C']).ravel()[1] for step in path_steps]
            torso_line.set_data(torso_x, torso_y)
            
            # Update foot paths
            left_x = [np.array(step['L']).ravel()[0] for step in path_steps]
            left_y = [np.array(step['L']).ravel()[1] for step in path_steps]
            left_line.set_data(left_x, left_y)
            
            right_x = [np.array(step['R']).ravel()[0] for step in path_steps]
            right_y = [np.array(step['R']).ravel()[1] for step in path_steps]
            right_line.set_data(right_x, right_y)
            
            # Update ZMP trajectory if provided
            if zmp_trajectory is not None and zmp_line is not None:
                # Filter ZMP points up to current time
                current_zmp_points = [point for point in zmp_trajectory if point[0] <= t]
                if current_zmp_points:
                    zmp_x = [point[1] for point in current_zmp_points]  # x-coordinate
                    zmp_y = [point[2] for point in current_zmp_points]  # y-coordinate
                    zmp_line.set_data(zmp_x, zmp_y)
        
        return tuple(return_elements)
    
    # Create animation
    frames = int(max_time * 100) + 1  # 100 frames per second of simulation
    num_frames = frames
    
    try:
        anim = animation.FuncAnimation(fig, animate, frames=frames, init_func=init, blit=True, interval=20)
        
        # Save animation
        writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(filename, writer=writer)
        
        plt.close(fig)  # Close the figure to free up memory
        print(f"Animation saved to {filename}")
    except Exception as e:
        print(f"Could not create animation: {str(e)}")
        plt.close(fig)  # Close the figure to free up memory
        
    return filename

# Function to plot ZMP trajectory over time
def plot_zmp_trajectory_over_time(zmp_trajectory, filename=None):
    """
    Plot the ZMP trajectory as position over time.
    
    Args:
        zmp_trajectory: List of [time, x, y] points defining the ZMP trajectory
        filename: Optional filename to save the plot
    """
    if not zmp_trajectory or len(zmp_trajectory) == 0:
        print("No ZMP trajectory data to plot")
        return
    
    # Create figure with two subplots (one for x, one for y)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Extract time, x, and y data
    time_data = [point[0] for point in zmp_trajectory]
    x_data = [point[1] for point in zmp_trajectory]
    y_data = [point[2] for point in zmp_trajectory]
    
    # Plot X position over time
    ax1.plot(time_data, x_data, 'b-', linewidth=2)
    ax1.set_ylabel('X Position (m)')
    ax1.set_title('ZMP X Position Over Time')
    ax1.grid(True)
    
    # Plot Y position over time
    ax2.plot(time_data, y_data, 'r-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('ZMP Y Position Over Time')
    ax2.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if filename is provided
    if filename:
        plt.savefig(filename)
        print(f"Saved ZMP trajectory time plot to {filename}")
    
    plt.close(fig)
    return filename

# Function to plot CoM trajectory over time
def plot_com_trajectory_over_time(com_trajectory, filename=None):
    """
    Plot the Center of Mass (CoM) trajectory as position over time.
    
    Args:
        com_trajectory: List of [time, x, y] points defining the CoM trajectory
        filename: Optional filename to save the plot
    """
    if not com_trajectory or len(com_trajectory) == 0:
        print("No CoM trajectory data to plot")
        return
    
    # Create figure with two subplots (one for x, one for y)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Extract time, x, and y data
    time_data = [point[0] for point in com_trajectory]
    x_data = [point[1] for point in com_trajectory]
    y_data = [point[2] for point in com_trajectory]
    
    # Plot X position over time
    ax1.plot(time_data, x_data, 'g-', linewidth=2)
    ax1.set_ylabel('X Position (m)')
    ax1.set_title('CoM X Position Over Time')
    ax1.grid(True)
    
    # Plot Y position over time
    ax2.plot(time_data, y_data, 'm-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('CoM Y Position Over Time')
    ax2.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if filename is provided
    if filename:
        plt.savefig(filename)
        print(f"Saved CoM trajectory time plot to {filename}")
    
    plt.close(fig)
    return filename

# Function to plot combined ZMP and CoM trajectories over time
def plot_combined_trajectories_over_time(zmp_trajectory, com_trajectory, filename=None):
    """
    Plot both ZMP and CoM trajectories as position over time on the same graph.
    
    Args:
        zmp_trajectory: List of [time, x, y] points defining the ZMP trajectory
        com_trajectory: List of [time, x, y] points defining the CoM trajectory
        filename: Optional filename to save the plot
    """
    if not zmp_trajectory or not com_trajectory or len(zmp_trajectory) == 0 or len(com_trajectory) == 0:
        print("No trajectory data to plot")
        return
    
    # Create figure with two subplots (one for x, one for y)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Extract ZMP data
    zmp_time = [point[0] for point in zmp_trajectory]
    zmp_x = [point[1] for point in zmp_trajectory]
    zmp_y = [point[2] for point in zmp_trajectory]
    
    # Extract CoM data
    com_time = [point[0] for point in com_trajectory]
    com_x = [point[1] for point in com_trajectory]
    com_y = [point[2] for point in com_trajectory]
    
    # Plot X positions over time
    ax1.plot(zmp_time, zmp_x, 'b-', linewidth=2, label='ZMP')
    ax1.plot(com_time, com_x, 'g-', linewidth=2, label='CoM')
    ax1.set_ylabel('X Position (m)')
    ax1.set_title('X Position Over Time')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Y positions over time
    ax2.plot(zmp_time, zmp_y, 'r-', linewidth=2, label='ZMP')
    ax2.plot(com_time, com_y, 'm-', linewidth=2, label='CoM')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Y Position Over Time')
    ax2.grid(True)
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if filename is provided
    if filename:
        plt.savefig(filename)
        print(f"Saved combined trajectory time plot to {filename}")
    
    plt.close(fig)
    return filename

# Run the demo for each velocity command
for i, vel_cmd in enumerate(velocity_commands):
    # Initial conditions
    # Define the initial positions in global frame
    global_frame = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
    initial_left_foot = np.array([0.0, 0.044, 0.0]).reshape(3, 1)  # Left foot slightly to the left
    initial_right_foot = np.array([0.0, -0.044, 0.0]).reshape(3, 1)  # Right foot slightly to the right
    initial_torso = np.array([0.0, 0.0, 0.0]).reshape(3, 1)  # Torso at origin
    
    # Set up the initial conditions for the planner
    # We'll start with right leg as support, so left leg will be swing
    next_support_leg = SupportLeg.RIGHT
    
    # Get current support and swing foot based on next_support_leg
    current_support = initial_right_foot
    current_swing = initial_left_foot

    current_swing = planner.transform_from_frame(current_swing, current_support)
    initial_torso = planner.transform_from_frame(initial_torso, current_support)

    print(f"Positions relative to support foot:")
    print(f"  Support foot: {current_support.flatten()}")
    print(f"  Swing foot w.r.t support foot: {current_swing.flatten()}")
    print(f"  Torso w.r.t support foot: {initial_torso.flatten()}")

    # Calculate the footstep sequence using the new method
    step_sequence = planner.calculate_footsteps(
        vel_cmd, current_swing, np.zeros(3), initial_torso, next_support_leg
    )
    
    # Transform the step sequence to global frame
    global_step_sequence = transform_step_sequence_to_global(step_sequence, global_frame)
    
    # Generate ZMP trajectory for the footstep sequence using the global step sequence
    # This ensures the ZMP trajectory is already in the global frame
    zmp_trajectory = planner.generate_complete_zmp_trajectory(global_step_sequence)
    
    # Generate CoM trajectory for the footstep sequence
    com_trajectory = planner.generate_complete_com_trajectory(global_step_sequence)
    
    # Print some information
    print(f"\nVelocity Command: {vel_cmd}")
    print(f"Number of steps in sequence: {len(step_sequence)}")
    print(f"Duration: {step_sequence[-1]['time']:.2f} seconds")
    
    # Print the first few steps
    print("\nFootstep sequence (first 3 steps):")
    for j, step in enumerate(global_step_sequence[:5]):
        left_pos = np.array(step['L']).ravel()
        right_pos = np.array(step['R']).ravel()
        torso_pos = np.array(step['C']).ravel()
        support_leg = "LEFT" if step['SF'] == SupportLeg.LEFT else "RIGHT" \
            if step['SF'] == SupportLeg.RIGHT else "BOTH"
        
        print(f"  Step {j}: Time={step['time']:.2f}s, Support Leg={support_leg}")
        print(f"    Left Foot: ({left_pos[0]:.3f}, {left_pos[1]:.3f})")
        print(f"    Right Foot: ({right_pos[0]:.3f}, {right_pos[1]:.3f})")
        print(f"    Torso: ({torso_pos[0]:.3f}, {torso_pos[1]:.3f})")
        
        # Print ZMP trajectory for the first step if available
        if j == 0 and zmp_trajectory:
            print("\n  ZMP trajectory for first step (first 3 samples):")
            time_array, zmp_pos = zmp_trajectory[0]
            for k in range(min(3, len(time_array))):
                print(f"    Time: {time_array[k]:.3f}s, ZMP: ({zmp_pos[k][0]:.3f}, {zmp_pos[k][1]:.3f})")
        
        # Print CoM trajectory for the first step if available
        if j == 0 and com_trajectory:
            print("\n  CoM trajectory for first step (first 3 samples):")
            time_array, com_pos = com_trajectory[0]
            for k in range(min(3, len(time_array))):
                print(f"    Time: {time_array[k]:.3f}s, CoM: ({com_pos[k][0]:.3f}, {com_pos[k][1]:.3f})")
    
    # Only create plots if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        plot_title = f"Footstep Sequence for Velocity Command {vel_cmd}"
        
        # Prepare ZMP data for plotting
        flat_zmp_trajectory = []
        current_time_offset = 0.0
        
        for segment in zmp_trajectory:
            time_array, zmp_pos = segment
            
            # Adjust time to be continuous from previous segments
            for t, pos in zip(time_array, zmp_pos):
                continuous_time = current_time_offset + t
                flat_zmp_trajectory.append([continuous_time, pos[0], pos[1]])
            
            # Update time offset for the next segment
            if len(time_array) > 0:
                current_time_offset += time_array[-1]
        
        # Prepare CoM data for plotting
        flat_com_trajectory = []
        current_time_offset = 0.0
        
        for segment in com_trajectory:
            time_array, com_pos = segment
            
            # Adjust time to be continuous from previous segments
            for t, pos in zip(time_array, com_pos):
                continuous_time = current_time_offset + t
                flat_com_trajectory.append([continuous_time, pos[0], pos[1]])
            
            # Update time offset for the next segment
            if len(time_array) > 0:
                current_time_offset += time_array[-1]
        
        # Plot footstep sequence with ZMP trajectory
        plot_footstep_sequence(global_step_sequence, plot_title, flat_zmp_trajectory)
        plt.savefig(f"footstep_sequence_{i}.png")
        print(f"Saved footstep sequence plot to footstep_sequence_{i}.png")
        
        try:
            # Create an animation of the footstep sequence with ZMP trajectory
            create_footstep_animation(global_step_sequence, f"walking_animation_{i}.mp4", 
                                      plot_title, flat_zmp_trajectory)
            print(f"Saved footstep animation to walking_animation_{i}.mp4")
            
            # Create a plot of ZMP trajectory over time
            plot_zmp_trajectory_over_time(flat_zmp_trajectory, f"zmp_time_plot_{i}.png")
            print(f"Saved ZMP time plot to zmp_time_plot_{i}.png")
            
            # Create a plot of CoM trajectory over time
            plot_com_trajectory_over_time(flat_com_trajectory, f"com_time_plot_{i}.png")
            print(f"Saved CoM time plot to com_time_plot_{i}.png")
            
            # Create a combined plot of ZMP and CoM trajectories over time
            plot_combined_trajectories_over_time(flat_zmp_trajectory, flat_com_trajectory, 
                                               f"combined_time_plot_{i}.png")
            print(f"Saved combined ZMP and CoM time plot to combined_time_plot_{i}.png")
        except Exception as e:
            print(f"Could not create animation or plots: {e}")
