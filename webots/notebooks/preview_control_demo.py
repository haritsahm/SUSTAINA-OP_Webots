import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from omegaconf import OmegaConf

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..',
                             'controllers', 'omnidirectional_walk', 'walk_engines'))

# Import the FootStepPlanner and PreviewControl classes
from foot_step_planner import FootStepPlanner, SupportLeg  # noqa: E402
from preview_control import PreviewControl  # noqa: E402

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..',
                          'controllers', 'omnidirectional_walk', 'config.yaml')
config = OmegaConf.load(config_path)

# Create the planner object
planner = FootStepPlanner(config)

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
        # The support foot frame is at support_frame, so we need to add this offset
        # to all positions to get them in the global frame
        global_step['L'] = planner.transform_to_frame(step['L'], global_frame)
        global_step['R'] = planner.transform_to_frame(step['R'], global_frame)
        global_step['C'] = planner.transform_to_frame(step['C'], global_frame)
        global_step['next_L'] = planner.transform_to_frame(step['next_L'], global_frame)
        global_step['next_R'] = planner.transform_to_frame(step['next_R'], global_frame)
        global_step['next_C'] = planner.transform_to_frame(step['next_C'], global_frame)

        global_step_sequence.append(global_step)

    return global_step_sequence


def run_preview_control_demo(vel_cmd, visualize=False):
    """Run the preview control demo with the given velocity command.
    
    This function demonstrates the footstep planning and preview control system.
    It calculates a sequence of footsteps based on velocity commands, where
    calculations are performed relative to the current support foot, and then
    results are transformed to the global reference frame.
    
    The key concept is that footstep planning happens in the support foot's
    reference frame, but for visualization and further processing, we transform
    the results back to the global reference frame. This approach ensures
    consistency across planning cycles and simplifies visualization.
    
    Parameters
    ----------
    vel_cmd : tuple or list
        Velocity command as (x_vel, y_vel, angular_vel)
    visualize : bool, optional
        Whether to visualize the results, by default False
    """
    print(f"Running preview control demo with velocity command: {vel_cmd}")
    
    # Initialize the planner
    planner = FootStepPlanner(config)
    pc = PreviewControl(config)
    
    # Initial conditions
    global_reference_frame = np.array([0.0, 0.0, 0.0]).reshape(3, 1)  # Global reference frame at origin
    
    # Define initial positions relative to the support foot (right foot)
    # These positions are already in the support foot's reference frame
    initial_right_foot = np.array([0.0, 0.0, 0.0]).reshape(3, 1)  # Support foot at origin
    initial_left_foot = np.array([0.0, 0.088, 0.0]).reshape(3, 1)  # Left foot (swing foot)
    initial_torso = np.array([0, 0.044, 0.0]).reshape(3, 1)  # Torso position

    # Initialize states
    state_x = pc.init_state_err(pos=0.0, vel=0.0, accel=0.0, e=0.0)
    state_y = pc.init_state_err(pos=0.044, vel=0.0, accel=0.0, e=0.0)

    # Start with right foot as support, so left foot will be swing
    next_support_leg = SupportLeg.RIGHT
    
    # Set current support and swing foot based on next_support_leg
    # Since we've defined positions relative to right foot as support,
    # we maintain this relationship when assigning current_support and current_swing
    if next_support_leg == SupportLeg.LEFT:  # Left foot is support
        current_support = initial_left_foot.copy()
        current_swing = initial_right_foot.copy()
    else:  # Right foot is support (our initial setup)
        current_support = initial_right_foot.copy()  # Right foot at origin
        current_swing = initial_left_foot.copy()  # Left foot relative to right foot
    current_torso = initial_torso.copy()

    # Initialize lists to store results
    all_left_foot_positions = []
    all_right_foot_positions = []
    all_torso_positions = []  # Track torso positions
    all_zmp_positions = []  # [time, x, y]
    all_com_positions = []  # [time, x, y]
    
    # Print initial positions
    print("Initial positions (global frame):")
    print(f"  Support foot: {current_support.flatten()}")
    print(f"  Swing foot: {current_swing.flatten()}")
    print(f"  Torso: {current_torso.flatten()}")
    print(f"  Left is swing: {next_support_leg == SupportLeg.RIGHT}")
    
    # Store initial foot and torso positions in global frame for visualization
    # We need to transform from local support foot frame to global frame
    if next_support_leg == SupportLeg.LEFT:  # Left foot is support
        all_right_foot_positions.append(planner.transform_to_frame(current_swing, global_reference_frame).flatten()[:2])
        all_left_foot_positions.append(planner.transform_to_frame(current_support, global_reference_frame).flatten()[:2])
    else:  # Right foot is support
        all_left_foot_positions.append(planner.transform_to_frame(current_swing, global_reference_frame).flatten()[:2])
        all_right_foot_positions.append(planner.transform_to_frame(current_support, global_reference_frame).flatten()[:2])
    
    # Store torso position
    all_torso_positions.append(planner.transform_to_frame(current_torso, global_reference_frame).flatten()[:2])

    # Run the preview control for a few cycles
    num_cycles = 4
    for cycle in range(num_cycles):
        print(f"Planning cycle {cycle+1}/{num_cycles}")
            
        # Print positions in the support foot's reference frame
        # Note: current_support should be [0,0,0] if we're consistently working in support foot frame
        print(f"Positions relative to support foot:")
        print(f"  Support foot: {current_support.flatten()}")
        print(f"  Swing foot w.r.t support foot: {current_swing.flatten()}")
        print(f"  Torso w.r.t support foot: {current_torso.flatten()}")
        
        # Calculate footsteps based on velocity command
        # All inputs are already in the support foot's reference frame
        # The calculate_footsteps method expects positions relative to the support foot
        # Note: In this implementation, current_swing and current_torso are relative to current_support
        # in the global frame, which means current_support is not at [0,0,0] but at its global position
        step_sequence = planner.calculate_footsteps(
            vel_cmd, 
            current_swing, 
            current_support,  # Should be zeros if we're consistently in support foot frame
            current_torso, 
            next_support_leg
        )
        
        # Transform step sequence from support foot frame back to global frame
        # This ensures all positions are in a consistent global reference frame for visualization
        global_step_sequence = transform_step_sequence_to_global(step_sequence, global_reference_frame)

        # This ensures the ZMP trajectory is already in the global frame
        global_zmp_trajectory = planner.generate_complete_zmp_trajectory(global_step_sequence)
        
        # Extract only the first segment of ZMP trajectory points for visualization
        first_segment_time, first_segment_zmp = global_zmp_trajectory[0]
        
        # Convert to numpy array for processing
        first_segment_zmp = np.array(first_segment_zmp).reshape(-1, 2)
        first_segment_time = np.array(first_segment_time) + cycle * planner.t_step
        for i in range(len(first_segment_zmp)):
            all_zmp_positions.append([first_segment_time[i], first_segment_zmp[i, 0], first_segment_zmp[i, 1]])

        # Simulate CoM trajectory using preview control
        total_steps = int(planner.t_step / planner.dt)

        for step in range(total_steps):
            # Get ZMP reference for current step

            # Update CoM state based on ZMP reference
            # update_state_err returns: (state_tuple, current_zmp, control_input)
            state_x, zmp_x, _ = pc.update_state_err(state_x, first_segment_zmp[:, 0])
            state_y, zmp_y, _ = pc.update_state_err(state_y, first_segment_zmp[:, 1])
            
            # Calculate current time for this CoM position
            step_time = cycle * planner.t_step + step * planner.dt
            
            # Extract position from state vector (first element of the state tuple)
            com_x = state_x[0][0]  # First element of state matrix is position
            com_y = state_y[0][0]  # First element of state matrix is position
            
            # Store CoM position with time for visualization [time, x, y]
            all_com_positions.append([step_time, float(com_x), float(com_y)])
            # all_zmp_positions.append([step_time, float(zmp_x), float(zmp_y)])
            
            # Move to next ZMP reference point
            first_segment_zmp = first_segment_zmp[1:]

        # Store foot positions for visualization
        # Update for next planning cycle
        if len(global_step_sequence) > 0:
            # Get the first step from the sequence
            first_step = global_step_sequence[0]

            # Debug: Print the next positions
            print(f"Updating positions for next cycle:")
            print(f"  Current support: {current_support.flatten()}")
            print(f"  Current swing: {current_swing.flatten()}")
            print(f"  Current torso: {current_torso.flatten()}")
            

            # Update positions for the next planning cycle based on the first planned step
            # The support leg alternates with each step
            next_support_leg = SupportLeg.LEFT if first_step['SF'] == SupportLeg.RIGHT else SupportLeg.RIGHT

            # Extract next foot positions from the step sequence
            # The footstep outputs from global_step_sequence are already transformed to the global reference frame
            # The calculations were done relative to the support foot, but transform_step_sequence_to_global
            # converted them to global coordinates
            if next_support_leg == SupportLeg.LEFT:  # Left foot will be support
                current_swing = np.array(first_step['next_R']).reshape(3, 1)  # Right foot will be swing
                current_support = np.array(first_step['next_L']).reshape(3, 1)  # Left foot will be support
                # Store positions for visualization (already in global frame coordinates)
                all_left_foot_positions.append(current_support.flatten()[:2])
                all_right_foot_positions.append(current_swing.flatten()[:2])
            else:  # RIGHT support
                current_swing = np.array(first_step['next_L']).reshape(3, 1)  # Left foot will be swing
                current_support = np.array(first_step['next_R']).reshape(3, 1)  # Right foot will be support
                # Store positions for visualization (already in global frame coordinates)
                all_right_foot_positions.append(current_support.flatten()[:2])
                all_left_foot_positions.append(current_swing.flatten()[:2])

            # Update torso position for next cycle (also in global frame coordinates)
            current_torso = np.array(first_step['next_C']).reshape(3, 1)
            
            # Store torso position for visualization (already in global frame coordinates)
            all_torso_positions.append(current_torso.flatten()[:2])

            # Print the updated positions (all in global frame coordinates)
            # These positions are in the global reference frame after being transformed
            # from the support foot's local frame
            print(f"  Next support: {current_support.flatten()}")
            print(f"  Next swing: {current_swing.flatten()}")
            print(f"  Next torso: {current_torso.flatten()}")
            print(f"  Next support leg: {'LEFT' if next_support_leg == SupportLeg.LEFT else 'RIGHT'}")
            print(f"  Left is swing: {next_support_leg == SupportLeg.RIGHT}")

    # Print final foot positions
    print("\nFinal positions:")
    print(f"  Left foot positions: {len(all_left_foot_positions)}")
    print(f"  Right foot positions: {len(all_right_foot_positions)}")

    # Return the results
    return {
        'left_foot_positions': all_left_foot_positions,
        'right_foot_positions': all_right_foot_positions,
        'torso_positions': all_torso_positions,  # Add torso positions to results
        'zmp_positions': all_zmp_positions,
        'com_positions': all_com_positions,
        'velocity_command': vel_cmd
    }


# Visualize the results of the preview control demo
def visualize_footsteps(all_left_foot_positions, all_right_foot_positions, torso_positions=None, zmp_positions=None, com_positions=None, vel_cmd=None):
    """
    Visualize the footstep planning results, including ZMP and CoM trajectories.
    
    Parameters:
    -----------
    all_left_foot_positions : list
        List of left foot positions (x, y)
    all_right_foot_positions : list
        List of right foot positions (x, y)
    torso_positions : list, optional
        List of torso positions (x, y)
    zmp_positions : list, optional
        List of ZMP positions with time [time, x, y]
    com_positions : list, optional
        List of CoM positions with time [time, x, y]
    vel_cmd : tuple, optional
        Velocity command used for the simulation
    """
    # Convert lists to numpy arrays for easier manipulation
    left_positions = np.array(all_left_foot_positions)
    right_positions = np.array(all_right_foot_positions)
    
    # Create a figure for spatial visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define foot dimensions (approximate)
    foot_length = 0.08
    foot_width = 0.04
    
    # Plot footsteps
    for i, (left_pos, right_pos) in enumerate(zip(left_positions, right_positions)):
        # Left foot (blue)
        left_rect = Rectangle(
            (left_pos[0] - foot_length/2, left_pos[1] - foot_width/2),
            foot_length, foot_width, 
            color='blue', alpha=0.5, label='Left foot' if i == 0 else None
        )
        ax.add_patch(left_rect)
        
        # Right foot (red)
        right_rect = Rectangle(
            (right_pos[0] - foot_length/2, right_pos[1] - foot_width/2),
            foot_length, foot_width,
            color='red', alpha=0.5, label='Right foot' if i == 0 else None
        )
        ax.add_patch(right_rect)
        
        # Add step number (only for non-initial steps)
        if i > 0:
            ax.text(left_pos[0], left_pos[1], str(i), color='white', fontsize=8, 
                    ha='center', va='center', fontweight='bold')
            ax.text(right_pos[0], right_pos[1], str(i), color='white', fontsize=8, 
                    ha='center', va='center', fontweight='bold')
    
    # Connect footsteps with lines to show the sequence
    if len(left_positions) > 1:
        ax.plot(left_positions[:, 0], left_positions[:, 1], 'b--', alpha=0.3)
    if len(right_positions) > 1:
        ax.plot(right_positions[:, 0], right_positions[:, 1], 'r--', alpha=0.3)
    
    # Plot torso, ZMP and CoM trajectories
    if torso_positions is not None and len(torso_positions) > 0:
        torso_positions = np.array(torso_positions)
        ax.plot(torso_positions[:, 0], torso_positions[:, 1], 'k-', linewidth=1.5, alpha=0.7, label='Torso')
        # Add markers for torso positions
        ax.scatter(torso_positions[:, 0], torso_positions[:, 1], color='black', s=25, marker='o')
    
    if zmp_positions is not None and len(zmp_positions) > 0:
        zmp_positions = np.array(zmp_positions)
        ax.plot(zmp_positions[:, 1], zmp_positions[:, 2], 'g-', linewidth=1.5, alpha=0.7, label='ZMP')
        
    if com_positions is not None and len(com_positions) > 0:
        com_positions = np.array(com_positions)
        print(f"Got COM positions: {com_positions.shape}")
        ax.plot(com_positions[:, 1], com_positions[:, 2], 'm-', linewidth=1.5, alpha=0.7, label='CoM')
        
    # Add a velocity vector at the start
    start_x = (left_positions[0][0] + right_positions[0][0]) / 2
    start_y = (left_positions[0][1] + right_positions[0][1]) / 2
    if vel_cmd is not None:
        ax.arrow(start_x, start_y, vel_cmd[0]*0.2, vel_cmd[1]*0.2, 
                head_width=0.02, head_length=0.03, fc='green', ec='green', label='Velocity')
    
    # Set axis properties
    ax.set_title(f'Footstep Planning (Velocity: {vel_cmd})')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')  # Equal aspect ratio
    
    # Add a time-based plot if we have trajectory data
    if (zmp_positions is not None and len(zmp_positions) > 0) or \
       (com_positions is not None and len(com_positions) > 0):
        # Create a second figure for time-based plots
        fig_time, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig_time.suptitle('ZMP and CoM Trajectories over Time')
        
        # Plot X position over time
        if zmp_positions is not None and len(zmp_positions) > 0:
            ax_x.plot(zmp_positions[:, 0], zmp_positions[:, 1], 'g-', linewidth=1.5, alpha=0.7, label='ZMP X')
        if com_positions is not None and len(com_positions) > 0:
            ax_x.plot(com_positions[:, 0], com_positions[:, 1], 'm-', linewidth=1.5, alpha=0.7, label='CoM X')
        ax_x.set_ylabel('X Position (m)')
        ax_x.legend()
        ax_x.grid(True, alpha=0.3)
        
        # Plot Y position over time
        if zmp_positions is not None and len(zmp_positions) > 0:
            ax_y.plot(zmp_positions[:, 0], zmp_positions[:, 2], 'g-', linewidth=1.5, alpha=0.7, label='ZMP Y')
        if com_positions is not None and len(com_positions) > 0:
            ax_y.plot(com_positions[:, 0], com_positions[:, 2], 'm-', linewidth=1.5, alpha=0.7, label='CoM Y')
        ax_y.set_xlabel('Time (s)')
        ax_y.set_ylabel('Y Position (m)')
        ax_y.legend()
        ax_y.grid(True, alpha=0.3)
        plt.tight_layout()
        
    
    # Adjust axis limits to include all footsteps with some margin
    all_x = np.concatenate([left_positions[:, 0], right_positions[:, 0]])
    all_y = np.concatenate([left_positions[:, 1], right_positions[:, 1]])
    
    # Include torso, ZMP and CoM points in axis limits if available
    if torso_positions is not None and len(torso_positions) > 0:
        all_x = np.append(all_x, torso_positions[:, 0])
        all_y = np.append(all_y, torso_positions[:, 1])
    if zmp_positions is not None and len(zmp_positions) > 0:
        all_x = np.append(all_x, zmp_positions[:, 1])
        all_y = np.append(all_y, zmp_positions[:, 2])
    if com_positions is not None and len(com_positions) > 0:
        all_x = np.append(all_x, com_positions[:, 1])
        all_y = np.append(all_y, com_positions[:, 2])
    
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    margin = 0.1
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    
    # Show the plots
    plt.tight_layout()
    plt.show()


# Run the demo with different velocity commands
def main():
    # Define velocity commands to test
    velocity_commands = [
        (0.5, 0.0, 0.0),   # Forward walking
        (0.0, 0.3, 0.0),    # Sideways walking
        (0.3, 0.2, 0.1),   # Diagonal walking with rotation
        (0.0, 0.0, 0.2),     # Turning in place
    ]
   
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run preview control demo')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    args = parser.parse_args()
   
    # Run the demo for each velocity command
    results = []
    for i, vel_cmd in enumerate(velocity_commands):
        result = run_preview_control_demo(vel_cmd)
       
        # Visualize the results if requested
        if result is not None and args.visualize:
            visualize_footsteps(
                all_left_foot_positions=result['left_foot_positions'], 
                all_right_foot_positions=result['right_foot_positions'],
                torso_positions=result['torso_positions'],
                zmp_positions=result['zmp_positions'],
                com_positions=result['com_positions'],
                vel_cmd=vel_cmd
            )
        results.append(result)

if __name__ == '__main__':
    main()