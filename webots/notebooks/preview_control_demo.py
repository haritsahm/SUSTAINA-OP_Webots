"""Demonstration of N-step generation with Preview Control for CoM trajectory generation.

This script demonstrates the generation of N consecutive footsteps for a bipedal robot
using a velocity-based step generator, with preview control for CoM trajectory generation.
The planner works by iteratively calculating the next torso and swing foot positions
relative to the current support foot, while the preview controller generates smooth
CoM trajectories that follow the ZMP references.

Reference: 
- Kajita et al. - Biped Walking Pattern Generation by using Preview Control of 
  Zero-Moment Point
- Maximo, Marcos - Omnidirectional ZMP-Based Walking for Humanoid
"""

import os
import sys
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                            'controllers', 'omnidirectional_walk', 'walk_engines')))

# Import the FootStepPlanner and PreviewControl classes
from foot_step_planner import FootStepPlanner, SupportLeg
from preview_control import PreviewControl

# Load configuration
config_path = os.path.join(
    os.path.dirname(__file__), '..',
    'controllers', 'omnidirectional_walk', 'config.yaml'
)
config = OmegaConf.load(config_path)

# Create the planner and preview controller
planner = FootStepPlanner(config)
preview_controller = PreviewControl(config)

def generate_n_steps_with_preview(
    n_steps: int,
    n_preview_steps: int,
    velocity_command: Tuple[float, float, float],
    initial_torso: np.ndarray,
    initial_left_foot: np.ndarray,
    initial_right_foot: np.ndarray,
    start_with_right: bool = True) -> List[Dict[str, np.ndarray]]:
    """Generate N footsteps with preview control for CoM trajectory generation.

    This function generates a sequence of footsteps based on a velocity command,
    using preview control to generate smooth CoM trajectories. For each step,
    it generates a preview window of N future steps to provide ZMP references
    for the preview controller.

    Parameters
    ----------
    n_steps : int
        Number of steps to generate
    n_preview_steps : int
        Number of future steps to preview for ZMP reference generation
    velocity_command : Tuple[float, float, float]
        Desired velocity command (vx, vy, omega) in m/s and rad/s
    initial_torso : np.ndarray, shape (3,1)
        Initial torso pose [x, y, theta] in global frame
    initial_left_foot : np.ndarray, shape (3,1)
        Initial left foot pose [x, y, theta] in global frame
    initial_right_foot : np.ndarray, shape (3,1)
        Initial right foot pose [x, y, theta] in global frame
    start_with_right : bool, optional
        Whether to start with right foot as support, by default True

    Returns
    -------
    List[Dict[str, np.ndarray]]
        List of step data dictionaries containing:
        - 'L', 'R': Left and right foot poses (3,1)
        - 'C': Torso pose (3,1)
        - 'SF': Support foot enum
        - 'next_L', 'next_R', 'next_C': Next poses
        - 'next_SF': Next support foot
        - 'com_trajectory': List of CoM trajectory points with:
            - 'time': Time in seconds
            - 'phase': Phase in [0,1]
            - 'position': CoM position (2,)
            - 'zmp_position': ZMP position (2,)
    """
    """
    Generate a sequence of footsteps with CoM trajectories using preview control.

    Parameters
    ----------
    n_steps : int
        Number of steps to actually generate and return
    n_preview_steps : int
        Number of future steps to use for preview control (preview window)
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
        - com_trajectory: List of CoM positions with time and phase
        - zmp_trajectory: List of ZMP positions with time and phase
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
    
    # Initialize preview control states
    state_x = preview_controller.init_state_err(pos=0.0, vel=0.0, accel=0.0, e=0.0)
    state_y = preview_controller.init_state_err(pos=0.0, vel=0.0, accel=0.0, e=0.0)

    num_steps = int(planner.t_step / planner.dt) + 1

    for step in range(n_steps):
        # Generate preview window of steps
        preview_steps = []
        preview_torso = current_torso.copy()
        preview_left = current_left.copy()
        preview_right = current_right.copy()
        preview_support = next_support_leg

        all_zmp_trajectory = []
        
        # Generate N preview steps including the current step
        for preview_idx in range(n_preview_steps):
            preview_step = planner.calculate_single_step(
                vel_cmd=vel_cmd,
                left_foot=preview_left,
                right_foot=preview_right,
                current_torso=preview_torso,
                next_support_leg=preview_support,
            )
            prev_zmp_step = planner.generate_complete_zmp_trajectory(preview_step)
            all_zmp_trajectory.extend([zmp['position'] for zmp in prev_zmp_step])
            preview_steps.append(preview_step)
            
            # Update poses for next preview step
            preview_left = preview_step['next_L']
            preview_right = preview_step['next_R']
            preview_torso = preview_step['next_C']
            preview_support = preview_step['next_SF']
        
        # Get the current step (first in preview window)
        step_data = preview_steps[0]
        
        # Extract ZMP references for x and y
        all_zmp_trajectory = np.asarray(all_zmp_trajectory).reshape(-1, 2)
        zmp_x = all_zmp_trajectory[:, 0]
        zmp_y = all_zmp_trajectory[:, 1]
        
        # Generate CoM trajectory using preview control
        com_trajectory = []
        for i in range(num_steps):
            # Update CoM state using preview control with future ZMP references
            state_x, zmp_x_current, _ = preview_controller.update_state_err(
                state_x, 
                zmp_x[i:],  # Remaining ZMP trajectory as preview window
            )
            state_y, zmp_y_current, _ = preview_controller.update_state_err(
                state_y, 
                zmp_y[i:],  # Remaining ZMP trajectory as preview window
            )
            
            # Store CoM position with time
            com_trajectory.append({
                'time': float(i * planner.dt),
                'phase': float(i * planner.dt / planner.t_step),
                'position': [float(state_x[0][0, 0]), float(state_y[0][0, 0])],
                'zmp_position': all_zmp_trajectory[i].tolist(),
            })
        
        step_data['com_trajectory'] = com_trajectory
        
        # Store step data
        step_sequence.append(step_data)
        
        # Update poses for next step
        current_left = step_data['next_L']
        current_right = step_data['next_R']
        current_torso = step_data['next_C']
        next_support_leg = step_data['next_SF']
    
    return step_sequence


def visualize_steps_with_preview(
    step_sequence: List[Dict[str, np.ndarray]],
    initial_poses: Dict[str, np.ndarray],
    show_plot: bool = True) -> None:
    """Visualize the footstep sequence with CoM and ZMP trajectories.

    Creates a figure with three subplots:
    1. Top: 2D view showing footsteps, torso path, CoM and ZMP trajectories
    2. Bottom left: X position vs time for CoM and ZMP
    3. Bottom right: Y position vs time for CoM and ZMP

    Parameters
    ----------
    step_sequence : List[Dict[str, np.ndarray]]
        List of step data dictionaries as returned by generate_n_steps_with_preview
    initial_poses : Dict[str, np.ndarray]
        Dictionary containing initial poses with keys:
        - 'L': Left foot pose [x, y, theta]
        - 'R': Right foot pose [x, y, theta]
        - 'C': Torso pose [x, y, theta]
    show_plot : bool, optional
        Whether to display the plot window, by default True

    Notes
    -----
    - Feet are visualized as rectangles with direction arrows
    - Torso positions shown as green dots with direction arrows
    - CoM trajectory in green, ZMP trajectory in magenta
    - Time plots show the progression of trajectories for each step
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
            # Get times and positions
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


def main() -> None:
    """Run the preview control demo with a forward walking velocity command.

    Demonstrates the generation of footsteps and CoM trajectories by:
    1. Setting initial foot and torso poses
    2. Generating 5 steps with a 5-step preview window
    3. Using a forward velocity command (0.1 m/s in x)
    4. Visualizing the resulting trajectories

    The demo shows how preview control generates smooth CoM trajectories
    that track the ZMP references while maintaining dynamic stability.
    """
    # Define test parameters
    N_STEPS = 5  # Number of steps to generate
    N_PREVIEW = 5  # Number of preview steps for ZMP planning
    VEL_CMD = (0.1, 0.2, 0.3)  # Forward walking at 0.1 m/s
    
    print(f"\nGenerating {N_STEPS} steps with {N_PREVIEW}-step preview window")
    print(f"Velocity command: {VEL_CMD}")
    print("-" * 60)
    
    # Initial poses
    initial_left = np.array([0.0, 0.054, 0.0]).reshape(3, 1)
    initial_right = np.array([0.0, -0.054, 0.0]).reshape(3, 1)
    initial_torso = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
    
    # Generate step sequence with preview control
    step_sequence = generate_n_steps_with_preview(
        n_steps=N_STEPS,
        n_preview_steps=N_PREVIEW,
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
        print(f"Trajectory points: {len(step_data['com_trajectory'])}")
    
    # Visualize the step sequence
    initial_poses = {
        'L': initial_left.ravel(),
        'R': initial_right.ravel(),
        'C': initial_torso.ravel()
    }
    visualize_steps_with_preview(step_sequence, initial_poses)


if __name__ == "__main__":
    main()