#!/usr/bin/env python3
"""Walk Engine Demo - Simulation of OpenZMPWalk.

This module provides a simulation environment for testing the OpenZMPWalk engine
without the need for a physical robot or Webots simulator. It extends OpenZMPWalk
to handle frame updates and trajectory generation for visualization purposes.
"""
import os
import sys
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from omegaconf import DictConfig, OmegaConf

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                            'controllers', 'omnidirectional_walk')))

# Import the FootStepPlanner and PreviewControl classes
from walk_engines.open_zmp_walk import OpenZMPWalk
from mathematics.linear_algebra import convert_quaternion_to_rpy, convert_quaternion_to_rotation


class OpenZMPWalkSim(OpenZMPWalk):
    """Simulation extension of OpenZMPWalk for testing and visualization.

    This class overrides key methods of OpenZMPWalk to work in a simulation
    environment, focusing on trajectory generation and frame updates without
    the need for actual robot hardware.
    """
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.swap_trigged = False

    def reset_frames(self, torso_pos: np.ndarray, torso_quat: np.ndarray,
                    left_foot_pos: np.ndarray, left_foot_quat: np.ndarray,
                    right_foot_pos: np.ndarray, right_foot_quat: np.ndarray) -> None:
        """Reset robot frames with given positions and orientations.

        Parameters
        ----------
        torso_pos : np.ndarray, shape (3,)
            Position [x, y, z] of torso in global frame
        torso_quat : np.ndarray, shape (4,)
            Orientation [qw, qx, qy, qz] of torso in global frame
        left_foot_pos : np.ndarray, shape (3,)
            Position of left foot in global frame
        left_foot_quat : np.ndarray, shape (4,)
            Orientation of left foot in global frame
        right_foot_pos : np.ndarray, shape (3,)
            Position of right foot in global frame
        right_foot_quat : np.ndarray, shape (4,)
            Orientation of right foot in global frame
        """
        # Update robot state containers (7D vectors)
        self.torso_robot[:3] = torso_pos
        self.torso_robot[3:] = torso_quat
        self.left_foot_robot[:3] = left_foot_pos
        self.left_foot_robot[3:] = left_foot_quat
        self.right_foot_robot[:3] = right_foot_pos
        self.right_foot_robot[3:] = right_foot_quat

        # Initialize FSP frames (x, y, theta)
        self.left_foot_fsp['initial'][:2] = left_foot_pos[:2]
        self.left_foot_fsp['initial'][2] = convert_quaternion_to_rpy(left_foot_quat)[2]
        self.right_foot_fsp['initial'][:2] = right_foot_pos[:2]
        self.right_foot_fsp['initial'][2] = convert_quaternion_to_rpy(right_foot_quat)[2]
        self.torso_fsp['initial'][:2] = torso_pos[:2]
        self.torso_fsp['initial'][2] = convert_quaternion_to_rpy(torso_quat)[2]

        # Initialize preview control state using the initial torso position
        self.state_x = self.pc.init_state_err(pos=float(self.torso_fsp['initial'][0]), e=0)
        self.state_y = self.pc.init_state_err(pos=float(self.torso_fsp['initial'][1]), e=0)

    def update_trajectories(self, com_xy: np.ndarray) -> None:
        """Update foot and torso trajectories based on current state.

        Generates smooth trajectories for swing foot, support foot, and torso
        using the current simulation time and target positions.

        Parameters
        ----------
        com_xy : np.ndarray, shape (2,)
            Current CoM position [x, y] from preview control
        """
        # Select appropriate foot frames based on swing state
        if self.left_is_swing:
            self.swing_foot_traj = self.calculate_swing_foot(
                self.t_sim, self.left_foot_fsp['initial'], self.left_foot_fsp['target'], rotation_as_quaternion=True)
            self.support_foot_traj = self.calculate_support_foot(
                self.t_sim, self.right_foot_fsp['initial'], self.right_foot_fsp['target'], rotation_as_quaternion=True)
        else:
            self.swing_foot_traj = self.calculate_swing_foot(
                self.t_sim, self.right_foot_fsp['initial'], self.right_foot_fsp['target'], rotation_as_quaternion=True)
            self.support_foot_traj = self.calculate_support_foot(
                self.t_sim, self.left_foot_fsp['initial'], self.left_foot_fsp['target'], rotation_as_quaternion=True)

        # Update torso trajectory
        self.torso_traj = self.calculate_com_trajectory(
            self.t_sim, com_xy, self.torso_fsp['initial'], self.torso_fsp['target'], rotation_as_quaternion=True)

    def update(self):
        """Generate N-step preview and update walking state.

        This method performs three main tasks:
        1. At step start (t_sim = 0):
           - Generates N preview steps using foot step planner
           - Computes complete ZMP trajectory for preview horizon
           - Updates target poses for current step
        2. During step execution:
           - Updates COM trajectory using preview control
           - Generates smooth foot trajectories
           - Maintains ZMP reference horizon
        3. At step completion:
           - Swaps support and swing feet
           - Updates step counter and timing
        """
        # Generate new preview sequence at step start
        if self.t_sim == 0:
            preview_support = self.next_support_leg
            preview_torso = self.torso_fsp['initial'].copy()
            preview_left = self.left_foot_fsp['initial'].copy()
            preview_right = self.right_foot_fsp['initial'].copy()
            
            preview_steps = []
            all_zmp_trajectory = []

            # Generate N preview steps in support foot frame
            for i in range(self.preview_steps):
                preview_step = self.fsp.calculate_single_step(
                    vel_cmd=self.cmd_vel,
                    left_foot=preview_left,
                    right_foot=preview_right,
                    current_torso=preview_torso,
                    next_support_leg=preview_support
                )
                preview_steps.append(preview_step)
                
                # Generate and store ZMP trajectory for this step
                zmp_step = self.fsp.generate_complete_zmp_trajectory(preview_step)
                all_zmp_trajectory.extend([zmp['position'] for zmp in zmp_step])
                
                # Update preview states for next step
                preview_left = preview_step['next_L']
                preview_right = preview_step['next_R']
                preview_torso = preview_step['next_C']
                preview_support = preview_step['next_SF']

            # Update target poses from first preview step
            self.left_foot_fsp['target'] = preview_steps[0]['next_L']
            self.right_foot_fsp['target'] = preview_steps[0]['next_R']
            self.torso_fsp['target'] = preview_steps[0]['next_C']
            self.next_support_leg = preview_steps[0]['next_SF']

            # Initialize ZMP preview horizon
            self.zmp_horizon = np.array(all_zmp_trajectory).reshape(-1, 2)[:self.pc.preview_horizon]

        # Update COM trajectory using preview control
        self.state_x, _, _ = self.pc.update_state_err(self.state_x, self.zmp_horizon[:, 0])
        self.state_y, _, _ = self.pc.update_state_err(self.state_y, self.zmp_horizon[:, 1])
        com_xy = [self.state_x[0][0].item(), self.state_y[0][0].item()]

        # Generate trajectories
        self.update_trajectories(com_xy)

        # Update simulation state
        self.t_sim += self.dt_sim
        self.current_zmp = self.zmp_horizon[0].copy()
        self.zmp_horizon = self.zmp_horizon[1:]

        # Handle step completion
        if self.t_sim >= self.fsp.t_step:
            self.swap_trigged = True
            self.t_sim = 0
            self.steps_count += 1
            self.swap_foot()


def create_frame_artists(ax):
    """Create artists for coordinate frame visualization.
    
    Args:
        ax: Matplotlib 3D axis
        
    Returns:
        tuple: Lists of line artists for each coordinate frame
    """
    # Create line artists for coordinate frames
    frame_length = 0.025  # Length of coordinate frame axes
    
    # Left foot frame
    left_frame = [
        ax.plot([], [], [], 'r-', linewidth=2)[0],  # x-axis
        ax.plot([], [], [], 'g-', linewidth=2)[0],  # y-axis
        ax.plot([], [], [], 'b-', linewidth=2)[0]   # z-axis
    ]
    
    # Right foot frame
    right_frame = [
        ax.plot([], [], [], 'r-', linewidth=2)[0],
        ax.plot([], [], [], 'g-', linewidth=2)[0],
        ax.plot([], [], [], 'b-', linewidth=2)[0]
    ]
    
    # Torso frame
    torso_frame = [
        ax.plot([], [], [], 'r-', linewidth=2)[0],
        ax.plot([], [], [], 'g-', linewidth=2)[0],
        ax.plot([], [], [], 'b-', linewidth=2)[0]
    ]
    
    return left_frame, right_frame, torso_frame, frame_length

def update_coordinate_frame(frame_artists, pose, frame_length):
    """Update coordinate frame visualization with orientation.
    
    Args:
        frame_artists: List of line artists for the frame
        pose: Pose array [x, y, z, qw, qx, qy, qz]
        frame_length: Length of coordinate frame axes
    """
    position = pose[:3]
    quaternion = pose[3:]
    
    # Convert quaternion to rotation matrix
    rotation = convert_quaternion_to_rotation(quaternion)
    
    # Create coordinate axes in local frame
    origin = np.zeros(3)
    x_axis = np.array([frame_length, 0, 0])
    y_axis = np.array([0, frame_length, 0])
    z_axis = np.array([0, 0, frame_length])
    
    # Rotate axes to global frame
    x_axis = rotation @ x_axis
    y_axis = rotation @ y_axis
    z_axis = rotation @ z_axis
    
    # Create line segments from origin to rotated endpoints
    x_line = np.vstack((origin, x_axis))
    y_line = np.vstack((origin, y_axis))
    z_line = np.vstack((origin, z_axis))
    
    # Update frame axes with position offset
    frame_artists[0].set_data_3d(
        position[0] + x_line[:, 0], position[1] + x_line[:, 1], position[2] + x_line[:, 2])
    frame_artists[1].set_data_3d(
        position[0] + y_line[:, 0], position[1] + y_line[:, 1], position[2] + y_line[:, 2])
    frame_artists[2].set_data_3d(
        position[0] + z_line[:, 0], position[1] + z_line[:, 1], position[2] + z_line[:, 2])

def plot_3d_trajectory(left_foot_pos, right_foot_pos, torso_pos, zmp_points, dt_sim):
    """Create 3D visualization of walking trajectory with animation.
    
    Args:
        left_foot_pos (np.ndarray): Left foot positions (N, 3)
        right_foot_pos (np.ndarray): Right foot positions (N, 3)
        torso_pos (np.ndarray): Torso positions (N, 3)
        zmp_points (np.ndarray): ZMP trajectory points (N, 3)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot full trajectories with lower alpha for reference
    ax.plot3D(left_foot_pos[:, 0], left_foot_pos[:, 1], left_foot_pos[:, 2], 'r-', label='Left Foot', alpha=0.3)
    ax.plot3D(right_foot_pos[:, 0], right_foot_pos[:, 1], right_foot_pos[:, 2], 'b-', label='Right Foot', alpha=0.3)
    ax.plot3D(torso_pos[:, 0], torso_pos[:, 1], torso_pos[:, 2], 'g-', label='Torso', alpha=0.3)
    ax.plot3D(zmp_points[:, 0], zmp_points[:, 1], np.zeros_like(zmp_points[:, 0]), 'k--', label='ZMP', alpha=0.3)
    
    # Initialize coordinate frames
    left_frame, right_frame, torso_frame, frame_length = create_frame_artists(ax)
    
    # Initialize markers for current positions
    left_marker, = ax.plot([], [], [], 'ro', markersize=5, label='Current Left Foot')
    right_marker, = ax.plot([], [], [], 'bo', markersize=5, label='Current Right Foot')
    torso_marker, = ax.plot([], [], [], 'go', markersize=5, label='Current Torso')
    zmp_marker, = ax.plot([], [], [], 'ko', markersize=5, label='Current ZMP')
    
    # Trail lines to show recent history
    trail_length = 20
    left_trail, = ax.plot([], [], [], 'r-', linewidth=2)
    right_trail, = ax.plot([], [], [], 'b-', linewidth=2)
    torso_trail, = ax.plot([], [], [], 'g-', linewidth=2)
    zmp_trail, = ax.plot([], [], [], 'k-', linewidth=2)
    
    # Set axis limits with extra padding for coordinate frames
    padding = frame_length * 2
    ax.set_xlim([min(left_foot_pos[:, 0].min(), right_foot_pos[:, 0].min(), torso_pos[:, 0].min()) - padding,
                 max(left_foot_pos[:, 0].max(), right_foot_pos[:, 0].max(), torso_pos[:, 0].max()) + padding])
    ax.set_ylim([min(left_foot_pos[:, 1].min(), right_foot_pos[:, 1].min(), torso_pos[:, 1].min()) - padding,
                 max(left_foot_pos[:, 1].max(), right_foot_pos[:, 1].max(), torso_pos[:, 1].max()) + padding])
    ax.set_zlim([0, max(left_foot_pos[:, 2].max(), right_foot_pos[:, 2].max(), torso_pos[:, 2].max()) + padding])
    
    # Customize 3D plot
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Walking Trajectory with Coordinate Frames')
    
    # Add custom legend entries for coordinate frames
    ax.plot([], [], [], 'r-', label='X-axis')
    ax.plot([], [], [], 'g-', label='Y-axis')
    ax.plot([], [], [], 'b-', label='Z-axis')
    ax.legend()
    ax.grid(True)
    
    def update(frame):
        # Update coordinate frames
        update_coordinate_frame(left_frame, left_foot_pos[frame], frame_length)
        update_coordinate_frame(right_frame, right_foot_pos[frame], frame_length)
        update_coordinate_frame(torso_frame, torso_pos[frame], frame_length)
        
        # Update ZMP marker
        zmp_marker.set_data_3d([zmp_points[frame, 0]], [zmp_points[frame, 1]], [0])
        
        # Update trail lines (using only position data)
        start_idx = max(0, frame - trail_length)
        left_trail.set_data_3d(left_foot_pos[start_idx:frame+1, :3][:, 0],
                              left_foot_pos[start_idx:frame+1, :3][:, 1],
                              left_foot_pos[start_idx:frame+1, :3][:, 2])
        right_trail.set_data_3d(right_foot_pos[start_idx:frame+1, :3][:, 0],
                               right_foot_pos[start_idx:frame+1, :3][:, 1],
                               right_foot_pos[start_idx:frame+1, :3][:, 2])
        torso_trail.set_data_3d(torso_pos[start_idx:frame+1, :3][:, 0],
                               torso_pos[start_idx:frame+1, :3][:, 1],
                               torso_pos[start_idx:frame+1, :3][:, 2])
        zmp_trail.set_data_3d(zmp_points[start_idx:frame+1, 0],
                             zmp_points[start_idx:frame+1, 1],
                             np.zeros(frame - start_idx + 1))
        
        # Update marker positions
        left_marker.set_data_3d([left_foot_pos[frame, 0]], [left_foot_pos[frame, 1]], [left_foot_pos[frame, 2]])
        right_marker.set_data_3d([right_foot_pos[frame, 0]], [right_foot_pos[frame, 1]], [right_foot_pos[frame, 2]])
        torso_marker.set_data_3d([torso_pos[frame, 0]], [torso_pos[frame, 1]], [torso_pos[frame, 2]])
        
        return (*left_frame, *right_frame, *torso_frame, left_marker, right_marker, torso_marker, zmp_marker,
                left_trail, right_trail, torso_trail, zmp_trail)
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(left_foot_pos),
                                 interval=dt_sim * 1000, blit=True)
    
    plt.show()

def plot_time_series(time_points, left_foot_pos, right_foot_pos, torso_pos, zmp_points):
    """Create time series plots of walking trajectory.
    
    Args:
        time_points (np.ndarray): Time points (N,)
        left_foot_pos (np.ndarray): Left foot positions (N, 3)
        right_foot_pos (np.ndarray): Right foot positions (N, 3)
        torso_pos (np.ndarray): Torso positions (N, 3)
        zmp_points (np.ndarray): ZMP trajectory points (N, 3)
    """
    fig = plt.figure(figsize=(15, 10))
    
    # X position vs time
    ax1 = fig.add_subplot(311)
    ax1.plot(time_points, left_foot_pos[:, 0], 'r-', label='Left Foot')
    ax1.plot(time_points, right_foot_pos[:, 0], 'b-', label='Right Foot')
    ax1.plot(time_points, torso_pos[:, 0], 'g-', label='Torso')
    ax1.plot(time_points, zmp_points[:, 0], 'k--', label='ZMP')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('X [m]')
    ax1.set_title('Position vs Time')
    ax1.grid(True)
    ax1.legend()
    
    # Y position vs time
    ax2 = fig.add_subplot(312)
    ax2.plot(time_points, left_foot_pos[:, 1], 'r-', label='Left Foot')
    ax2.plot(time_points, right_foot_pos[:, 1], 'b-', label='Right Foot')
    ax2.plot(time_points, torso_pos[:, 1], 'g-', label='Torso')
    ax2.plot(time_points, zmp_points[:, 1], 'k--', label='ZMP')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Y [m]')
    ax2.grid(True)
    ax2.legend()
    
    # Z position vs time
    ax3 = fig.add_subplot(313)
    ax3.plot(time_points, left_foot_pos[:, 2], 'r-', label='Left Foot')
    ax3.plot(time_points, right_foot_pos[:, 2], 'b-', label='Right Foot')
    ax3.plot(time_points, torso_pos[:, 2], 'g-', label='Torso')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Z [m]')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print('Starting script...')
    # Load configuration
    config_path = os.path.join(
        os.path.dirname(__file__), '..',
        'controllers', 'omnidirectional_walk', 'config.yaml'
    )
    config = OmegaConf.load(config_path)

    walk_engine = OpenZMPWalkSim(config)
    print('Walk engine initialized')

    init_torso_pos = np.zeros(3)
    init_torso_quat = np.array([1, 0, 0, 0])
    init_left_foot_pos = np.array([0, walk_engine.foot_separation, 0])
    init_left_foot_quat = np.array([1, 0, 0, 0])
    init_right_foot_pos = np.array([0, -walk_engine.foot_separation, 0])
    init_right_foot_quat = np.array([1, 0, 0, 0])
    walk_engine.reset_frames(init_torso_pos, init_torso_quat, init_left_foot_pos, init_left_foot_quat, init_right_foot_pos, init_right_foot_quat)

    vel_cmd = (0.1, 0., 0.1)
    n_steps = 8

    walk_engine.set_walking_command(*vel_cmd)
    print('Walking command set')

    print('Initializing trajectory storage')
    # Initialize trajectory storage with left/right foot positions
    time_points = []
    left_foot_pos = []
    right_foot_pos = []
    torso_pos = []
    zmp_points = []
    
    total_sim_steps = n_steps * int(walk_engine.fsp.t_step / walk_engine.fsp.dt) + 1
    for i in range(total_sim_steps):
        walk_engine.update()

        # Use left_is_swing to determine which foot is which
        if walk_engine.swap_trigged:
            walk_engine.swap_trigged = False
            continue

        # Store current time
        curr_time = i * walk_engine.dt_sim
        time_points.append(curr_time)

        if walk_engine.left_is_swing:
            left_foot_pos.append(walk_engine.swing_foot_traj.copy())
            right_foot_pos.append(walk_engine.support_foot_traj.copy())
        else:
            left_foot_pos.append(walk_engine.support_foot_traj.copy())
            right_foot_pos.append(walk_engine.swing_foot_traj.copy())
        
        torso_pos.append(walk_engine.torso_traj.copy())
        zmp_points.append(walk_engine.current_zmp.copy())
    
    print('Converting to numpy arrays')
    # Convert to numpy arrays for plotting
    time_points = np.array(time_points)
    left_foot_pos = np.array(left_foot_pos)
    right_foot_pos = np.array(right_foot_pos)
    torso_pos = np.array(torso_pos)
    zmp_points = np.array(zmp_points)

    print(f"Got left foot pos: {left_foot_pos.shape}")
    print(f"Got right foot pos: {right_foot_pos.shape}")
    print(f"Got torso pos: {torso_pos.shape}")
    print(f"Got zmp points: {zmp_points.shape}")
    
    print('Creating plots')    
    # Plot time series
    plot_time_series(time_points, left_foot_pos, right_foot_pos, torso_pos, zmp_points)

    # Plot 3D trajectory
    plot_3d_trajectory(left_foot_pos, right_foot_pos, torso_pos, zmp_points, dt_sim=walk_engine.dt_sim)
