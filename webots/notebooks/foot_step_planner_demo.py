"""Demonstration of N-step generation using the FootStepPlanner.

This script demonstrates the generation of N consecutive footsteps for a bipedal robot
using a velocity-based step generator. The planner works by iteratively calculating
the next torso and swing foot positions relative to the current support foot.

Reference: Maximo, Marcos - Omnidirectional ZMP-Based Walking for Humanoid
Section 3.2 - Selecting Next Torso and Swing Foot Poses
"""

import os
import sys
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Add the parent directory to the path so we can import the foot_step_planner module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 
                                            'controllers', 'omnidirectional_walk', 'walk_engines')))

# Import the FootStepPlanner class
from foot_step_planner import FootStepPlanner, SupportLeg

# Load configuration
config_path = os.path.join(
    os.path.dirname(__file__), '..',
    'controllers', 'omnidirectional_walk', 'config.yaml'
)
config = OmegaConf.load(config_path)

# Create the planner
planner = FootStepPlanner(config)




def generate_n_steps(n_steps: int,
                     velocity_command: Tuple[float, float, float],
                     initial_torso: np.ndarray,
                     initial_left_foot: np.ndarray,
                     initial_right_foot: np.ndarray,
                     start_with_right: bool = True) -> List[Dict]:
    """Generate a sequence of footsteps with ZMP trajectories.

    Parameters
    ----------
    n_steps : int
        Number of steps to generate
    velocity_command : tuple
        Desired velocity as (v_x, v_y, omega) in m/s and rad/s
    initial_torso : ndarray, shape (3,)
        Initial torso pose [x, y, theta] in global frame
    initial_left_foot : ndarray, shape (3,)
        Initial left foot pose [x, y, theta] in global frame
    initial_right_foot : ndarray, shape (3,)
        Initial right foot pose [x, y, theta] in global frame
    start_with_right : bool, optional
        If True, start with right foot as support, by default True

    Returns
    -------
    list of dict
        List of step data, each containing:
        - SF: Support foot (LEFT/RIGHT)
        - L, R: Left/right foot poses [x, y, theta]
        - C: Current torso pose [x, y, theta]
        - next_L, next_R: Next foot poses [x, y, theta]
        - next_C: Next torso pose [x, y, theta]
        - next_SF: Next support foot
        - zmp_trajectory: List of ZMP positions
    """
    step_sequence = []
    
    # Initialize current poses
    current_torso = initial_torso.copy()
    current_left = initial_left_foot.copy()
    current_right = initial_right_foot.copy()
    
    # Initial support leg
    next_support_leg = SupportLeg.RIGHT if start_with_right else SupportLeg.LEFT
    
    # Apply velocity limits
    vel_cmd = planner.limit_velocity(velocity_command)
    
    for step in range(n_steps):
        # Generate single step data using planner
        step_data = planner.calculate_single_step(
            vel_cmd=vel_cmd,
            left_foot=current_left,
            right_foot=current_right,
            current_torso=current_torso,
            next_support_leg=next_support_leg,
        )
        
        # Generate CoM trajectory with ZMP positions
        com_trajectory = planner.generate_complete_com_trajectory(
            step_data=step_data,
            with_zmp=True
        )
        step_data['com_trajectory'] = com_trajectory

        # Store step data
        step_sequence.append(step_data)
        
        # Update poses for next step
        current_left = step_data['next_L']
        current_right = step_data['next_R']
        current_torso = step_data['next_C']
        next_support_leg = step_data['next_SF']
    
    return step_sequence


def visualize_steps(step_sequence, initial_poses, show_plot: bool = True):
    """Visualize the footstep sequence with torso trajectory and foot placements.
    
    Plots:
    - Initial foot positions
    - Each step's foot positions with direction arrows
    - Torso trajectory with direction arrows
    - ZMP and CoM trajectories color-coded by phase
    - X and Y trajectories over time
    
    Args:
        step_sequence: List of step data dictionaries
        initial_poses: Dictionary with initial L, R, C poses
        show_plot: Whether to display the plot
    """
    from matplotlib.patches import Rectangle
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[2, 1])
    
    # Top subplot: 2D trajectory view
    ax_2d = fig.add_subplot(gs[0, :])
    ax_2d.set_title('2D Step Sequence with CoM and ZMP Trajectories')
    ax_2d.set_xlabel('X [m]')
    ax_2d.set_ylabel('Y [m]')
    ax_2d.grid(True)
    
    # Bottom subplots: X and Y trajectories over time
    ax_x = fig.add_subplot(gs[1, 0])
    ax_x.set_title('X Position vs Time')
    ax_x.set_xlabel('Time [s]')
    ax_x.set_ylabel('X [m]')
    ax_x.grid(True)
    
    ax_y = fig.add_subplot(gs[1, 1])
    ax_y.set_title('Y Position vs Time')
    ax_y.set_xlabel('Time [s]')
    ax_y.set_ylabel('Y [m]')
    ax_y.grid(True)
    
    def plot_foot(ax, pose, is_left=True, alpha=1.0, label=None):
        """Plot a foot rectangle with direction arrow."""
        # Foot dimensions (approximate)
        length = 0.08  # meters
        width = 0.05   # meters
        
        # Create rectangle patch
        rect = Rectangle(
            xy=(pose[0] - length/2, pose[1] - width/2),
            width=length,
            height=width,
            angle=np.degrees(pose[2]),
            color='blue' if is_left else 'red',
            alpha=alpha,
            fill=False,
            linewidth=2,
            label=label
        )
        ax.add_patch(rect)
        
        # Add direction arrow
        arrow_length = length/2
        dx = arrow_length * np.cos(pose[2])
        dy = arrow_length * np.sin(pose[2])
        ax.arrow(
            pose[0], pose[1],
            dx, dy,
            head_width=0.02,
            head_length=0.02,
            fc='blue' if is_left else 'red',
            ec='blue' if is_left else 'red',
            alpha=alpha,
            linewidth=1.5
        )
        
        # Add label if provided
        if label:
            ax.text(pose[0], pose[1], label,
                   fontsize=8, ha='center', va='bottom',
                   color='blue' if is_left else 'red')
    
    def plot_torso(ax, pose, alpha=1.0, label=None):
        """Plot torso position with direction arrow."""
        ax.plot(pose[0], pose[1], 'go', alpha=alpha, label=label)
        
        # Add direction arrow
        arrow_length = 0.05
        dx = arrow_length * np.cos(pose[2])
        dy = arrow_length * np.sin(pose[2])
        ax.arrow(
            pose[0], pose[1],
            dx, dy,
            head_width=0.02,
            head_length=0.02,
            fc='green',
            ec='green',
            alpha=alpha,
            linewidth=1.5
        )
    
    # Plot initial poses
    plot_foot(ax_2d, initial_poses['L'], is_left=True, label='Initial Left')
    plot_foot(ax_2d, initial_poses['R'], is_left=False, label='Initial Right')
    plot_torso(ax_2d, initial_poses['C'], label='Initial Torso')
    
    # Time offset for trajectory plotting
    t_offset = 0.0
    
    # Plot each step
    for i, step in enumerate(step_sequence):
        # Plot feet and torso with reduced alpha
        plot_foot(ax_2d, step['L'], is_left=True, alpha=0.5)
        plot_foot(ax_2d, step['R'], is_left=False, alpha=0.5)
        plot_torso(ax_2d, step['C'], alpha=0.5)
        
        # Plot CoM and ZMP trajectories
        if 'com_trajectory' in step:
            times = [t_offset + point['time'] for point in step['com_trajectory']]
            com_x = [point['position'][0] for point in step['com_trajectory']]
            com_y = [point['position'][1] for point in step['com_trajectory']]
            zmp_x = [point['zmp_position'][0] for point in step['com_trajectory']]
            zmp_y = [point['zmp_position'][1] for point in step['com_trajectory']]
            
            # 2D trajectory plot
            ax_2d.plot(com_x, com_y, 'g-', alpha=0.5, label='CoM' if i == 0 else '')
            ax_2d.plot(zmp_x, zmp_y, 'm-', alpha=0.5, label='ZMP' if i == 0 else '')
            
            # X and Y trajectory plots
            ax_x.plot(times, com_x, 'g-', alpha=0.5)
            ax_x.plot(times, zmp_x, 'm-', alpha=0.5)
            ax_y.plot(times, com_y, 'g-', alpha=0.5)
            ax_y.plot(times, zmp_y, 'm-', alpha=0.5)
            
            # Update time offset for next step
            t_offset += step['com_trajectory'][-1]['time']
    
    # Add legends
    ax_2d.legend()
    
    # Set equal aspect ratio for 2D plot
    ax_2d.set_aspect('equal')
    
    # Add legends to trajectory plots
    ax_x.legend(['CoM X', 'ZMP X'])
    ax_y.legend(['CoM Y', 'ZMP Y'])
    
    # Adjust layout and display
    plt.tight_layout()
    if show_plot:
        plt.show()





def main():
    # Define test parameters
    N_STEPS = 5  # Number of steps to generate
    VEL_CMD = (0.1, 0.1, 0.2)  # Forward walking at 0.1 m/s
    
    print(f"\nGenerating {N_STEPS} steps with velocity command: {VEL_CMD}")
    print("-" * 60)
    
    # Initial poses
    initial_left = np.array([0.0, 0.054, 0.0]).reshape(3, 1)
    initial_right = np.array([0.0, -0.054, 0.0]).reshape(3, 1)
    initial_torso = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
    
    # Generate step sequence
    step_sequence = generate_n_steps(
        n_steps=N_STEPS,
        velocity_command=VEL_CMD,
        initial_torso=initial_torso,
        initial_left_foot=initial_left,
        initial_right_foot=initial_right,
        start_with_right=True
    )
    
    # Print step information
    for i, step_data in enumerate(step_sequence):
        print(f"\nStep {i+1}:")
        print(f"Support foot: {step_data['SF']}")
        print(f"Left foot: {step_data['L'].ravel()}")
        print(f"Right foot: {step_data['R'].ravel()}")
        print(f"Torso: {step_data['C'].ravel()}")
        print(f"CoM trajectory points: {len(step_data['com_trajectory'])}")
    
    # Visualize the step sequence
    initial_poses = {
        'L': initial_left.ravel(),
        'R': initial_right.ravel(),
        'C': initial_torso.ravel()
    }
    visualize_steps(step_sequence, initial_poses)

if __name__ == "__main__":
    main()
