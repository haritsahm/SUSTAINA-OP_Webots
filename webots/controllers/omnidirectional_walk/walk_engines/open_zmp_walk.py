import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'kinematics_dynamics'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mathematics'))

import numpy as np
from typing import List
from omegaconf import DictConfig
import logging
from .foot_step_planner import FootStepPlanner, SupportLeg
from .preview_control import PreviewControl
from kinematics_dynamics.kinematics_dynamics import KinematicsDynamics
from mathematics import convert_rotation_to_quaternion, convert_quaternion_to_rotation, get_rotation_z, get_rotation_rpy, convert_rotation_to_rpy

# Global constants for joint IDs
TORSO_ID = 0       # Base/torso joint ID
LEFT_FOOT_ID = 18  # Left foot joint ID
RIGHT_FOOT_ID = 17 # Right foot joint ID

logger = logging.getLogger(__name__)

class OpenZMPWalk:
    """
    Omnidirectional ZMP-Based Walking for a Humanoid Robot.
    
    This class implements a Zero Moment Point (ZMP) based walking engine
    that allows omnidirectional walking for humanoid robots. The implementation
    uses homogeneous transformation matrices to represent the robot's frames
    and performs footstep planning in the support foot frame.
    
    The walking engine follows these steps:
    1. Extract current robot state (torso, support foot, swing foot)
    2. Transform torso pose to support foot frame
    3. Plan footsteps and ZMP trajectory in support foot frame
    4. Transform results back to global frame
    5. Calculate COM and swing foot trajectories
    6. Apply inverse kinematics to update robot joint angles
    
    Based on ITAndroid team's approach from Maximo, M. R's master thesis.
    """
    
    def __init__(self, config: DictConfig):
        """Initialize the OpenZMPWalk engine.

        This class manages walking motion generation using preview control and
        foot step planning. It maintains three types of frame data:
        1. Foot step planner frames (FSP): Used for step planning, stores (x, y, theta)
        2. Robot state frames: Current robot state as 7D vectors (x, y, z, qw, qx, qy, qz)
        3. Trajectory frames: Generated motion as 7D vectors (x, y, z, qw, qx, qy, qz)

        Parameters
        ----------
        config : DictConfig
            Configuration parameters for the walking engine containing:
            - walk: Walking parameters (step frequency, height, etc.)
            - preview_control: Preview controller parameters
            - foot_step_planner: Footstep planner parameters

        Notes
        -----
        All frames are expressed in the global coordinate system (fixed at origin).
        The global frame is right-handed with:
        - X-axis: Forward direction
        - Y-axis: Left direction
        - Z-axis: Up direction
        """
        self.config = config
        
        # Initialize controllers
        self.pc = PreviewControl(config)
        self.fsp = FootStepPlanner(config)
        
        # Walking parameters from config
        self.step_height = config.walking.step_height         # meters
        self.foot_separation = config.body.foot_separation # meters
        self.com_height = config.body.com_height          # meters
        self.foot_offset = np.zeros(3, dtype=np.float32)     # meters, offset from ankle to foot sole
        
        # Walking command (in global frame)
        self.cmd_vel = np.zeros(3, dtype=np.float32)  # [vx, vy, omega] - m/s, m/s, rad/s
        
        # State variables
        self.t_sim = 0.0     # Current time within step cycle [0, t_step]
        self.dt_sim = config.preview_control.dt   # Simulation time dt (s)
        self.left_is_swing = True  # True if left foot is swing foot
        self.next_support_leg = SupportLeg.RIGHT if self.left_is_swing else SupportLeg.LEFT
        self.first_step = True      # True during initial step
        self.steps_count = 1        # Number of steps taken
        self.preview_steps = config.walking.preview_steps

        # Global frame (fixed at origin)
        self.global_frame = np.eye(4, dtype=np.float32)  # Right-handed: X forward, Y left, Z up

        # Frame containers for foot step planner (FSP)
        # Each frame stores (x, y, theta) for planning in global frame
        self.torso_fsp = {
            'initial': np.zeros(3, dtype=np.float32),  # Current torso pose for planning
            'target': np.zeros(3, dtype=np.float32)    # Next torso pose from planner
        }
        
        self.left_foot_fsp = {
            'initial': np.zeros(3, dtype=np.float32),  # Current support foot pose
            'target': np.zeros(3, dtype=np.float32)    # Next support foot pose
        }
        
        self.right_foot_fsp = {
            'initial': np.zeros(3, dtype=np.float32),  # Current swing foot pose
            'target': np.zeros(3, dtype=np.float32)    # Target swing foot landing pose
        }
        
        # Robot state containers (from kinematics)
        # Stored as 7D vectors (x, y, z, qw, qx, qy, qz) in global frame
        self.torso_robot = np.zeros(7, dtype=np.float32)       # Current torso state
        self.right_foot_robot = np.zeros(7, dtype=np.float32)  # Current right foot state
        self.left_foot_robot = np.zeros(7, dtype=np.float32)   # Current left foot state
        
        # Generated trajectory containers
        # Stored as 7D vectors (x, y, z, qw, qx, qy, qz) in global frame
        self.swing_foot_traj = np.zeros(7, dtype=np.float32)   # Swing foot trajectory
        self.support_foot_traj = np.zeros(7, dtype=np.float32) # Support foot trajectory
        self.torso_traj = np.zeros(7, dtype=np.float32)        # Torso trajectory

        # Preview control state variables
        self.zmp_horizon = np.zeros((self.pc.preview_horizon, 2), dtype=np.float32)  # Future ZMP references
        self.zmp_current = np.zeros(2, dtype=np.float32)                            # Current ZMP position [x, y]
        
    def reset_frames(self, robot_kd: KinematicsDynamics = None) -> None:
        """Reset all frames and states for walking initialization.

        This method performs three main tasks:
        1. Resets walking state variables (flags, counters, time)
        2. Updates robot state containers from current kinematics
        3. Initializes FSP frames and trajectories for first step

        The initialization assumes:
        - Right foot will swing first (left_is_swing = False)
        - Left foot is initial support foot
        - First step needs sway motion to shift CoM

        Parameters
        ----------
        robot_kd : KinematicsDynamics, optional
            Robot kinematics/dynamics interface. If None, uses default poses.

        Notes
        -----
        - All poses are in global coordinate system
        - Robot state uses 7D vectors (x, y, z, qw, qx, qy, qz)
        - FSP frames use (x, y, theta) for planning
        - Preview control states initialized at current torso position
        """
        # Initialize walking state
        self.left_is_swing = False  # Start with right leg swinging first
        self.first_step = True      # Flag for the first step (adds sway to move torso)

        # Reset step counter and simulation time
        self.t_sim = 0
        self.steps_count = 1

        # Update robot data containers with current robot state
        self.torso_robot[:3] = robot_kd.get_position(joint_id=TORSO_ID)    # Set translation vector (3x1)
        self.torso_robot[3:] = convert_rotation_to_quaternion(robot_kd.get_rotation(joint_id=TORSO_ID))  # Set rotation matrix (3x3)

        # Update robot data containers for feet
        self.left_foot_robot[:3] = robot_kd.get_position(LEFT_FOOT_ID)
        self.left_foot_robot[3:] = convert_rotation_to_quaternion(robot_kd.get_rotation(LEFT_FOOT_ID))
        self.right_foot_robot[:3] = robot_kd.get_position(RIGHT_FOOT_ID)
        self.right_foot_robot[3:] = convert_rotation_to_quaternion(robot_kd.get_rotation(RIGHT_FOOT_ID))
        
        # Initialize footstep planner frames from robot data
        # Since left_is_swing is False, the left foot is the initial support foot
        # This is technically the last foot that was swung, which will be set as the support foot
        self.support_foot_fsp['initial'][:2] = self.left_foot_robot[:2]
        self.support_foot_fsp['initial'][2] = convert_quaternion_to_rpy(self.left_foot_robot[3:])[2]
        self.swing_foot_fsp['initial'][:2] = self.right_foot_robot[:2]
        self.swing_foot_fsp['initial'][2] = convert_quaternion_to_rpy(self.right_foot_robot[3:])[2]
        self.torso_fsp['initial'][:2] = self.torso_robot[:2]
        self.torso_fsp['initial'][2] = convert_quaternion_to_rpy(self.torso_robot[3:])[2]

        # Set target positions to match initial positions (no movement initially)
        self.torso_fsp['target'] = self.torso_fsp['initial'].copy()
        self.support_foot_fsp['target'] = self.support_foot_fsp['initial'].copy()
        self.swing_foot_fsp['target'] = self.swing_foot_fsp['initial'].copy()

        # Initialize preview control state using the initial torso position
        self.state_x = self.pc.init_state_err(pos=float(self.torso_fsp['initial'][0]), e=0)
        self.state_y = self.pc.init_state_err(pos=float(self.torso_fsp['initial'][1]), e=0)
        
        # Initialize trajectory frames with initial positions
        self.torso_traj = self.torso_fsp['initial'][:2].ravel().tolist() + convert_rpy_to_quaternion(0, 0, self.torso_fsp['initial'][2]).tolist()
        self.support_foot_traj = self.support_foot_fsp['initial'][:2].ravel().tolist() + convert_rpy_to_quaternion(0, 0, self.support_foot_fsp['initial'][2]).tolist()
        self.swing_foot_traj = self.swing_foot_fsp['initial'][:2].ravel().tolist() + convert_rpy_to_quaternion(0, 0, self.swing_foot_fsp['initial'][2]).tolist()

    def _get_current_robot_state(self, robot_kd: KinematicsDynamics, next_supp_leg: SupportLeg) -> None:
        """Update robot state containers with current kinematics data.

        Retrieves current robot state from kinematics and updates the 7D state
        vectors (position + quaternion) for torso and feet in global frame.

        Parameters
        ----------
        robot_kd : KinematicsDynamics
            Interface to robot kinematics providing position and rotation data
        next_supp_leg : SupportLeg
            Next support leg (LEFT or RIGHT) for state tracking

        Notes
        -----
        - All positions and orientations are in global frame
        - State vectors format: [x, y, z, qw, qx, qy, qz]
        - Position from get_position(): [x, y, z]
        - Rotation converted to quaternion from rotation matrix
        """
        # Update torso state
        self.torso_robot[:3] = robot_kd.get_position(TORSO_ID)
        self.torso_robot[3:] = convert_rotation_to_quaternion(robot_kd.get_rotation(TORSO_ID))
        
        # Update feet states
        self.left_foot_robot[:3] = robot_kd.get_position(LEFT_FOOT_ID)
        self.left_foot_robot[3:] = convert_rotation_to_quaternion(robot_kd.get_rotation(LEFT_FOOT_ID))
        
        self.right_foot_robot[:3] = robot_kd.get_position(RIGHT_FOOT_ID)
        self.right_foot_robot[3:] = convert_rotation_to_quaternion(robot_kd.get_rotation(RIGHT_FOOT_ID))
        
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Current Robot State (global frame):")
            logging.debug(f"Torso: pos={self.torso_robot[:3]}, quat={self.torso_robot[3:]}")
            logging.debug(f"Left Foot: pos={self.left_foot_robot[:3]}, quat={self.left_foot_robot[3:]}")
            logging.debug(f"Right Foot: pos={self.right_foot_robot[:3]}, quat={self.right_foot_robot[3:]}")
    
    def set_walking_command(self, x_speed: float, y_speed: float, a_speed: float):
        """
        Set the walking command for the next step
        
        Args:
            x_speed: Forward/backward speed in m/s
            y_speed: Lateral speed in m/s
            a_speed: Angular speed in rad/s
        """
        # Set velocity command expected by the footstep planner
        self.cmd_vel = np.array([x_speed, y_speed, a_speed], dtype=np.float32).reshape((3, 1))
    
    def swap_foot(self) -> None:
        """Swap support and swing feet in the footstep planner frames.

        Updates the FSP frame containers and swing state based on the next
        support leg. The previous swing foot becomes the new support foot,
        and vice versa.

        Parameters
        ----------
        next_support_leg : SupportLeg
            Specifies which leg (LEFT/RIGHT) will be the support leg
            after the swap

        Notes
        -----
        - Updates left_is_swing flag based on next_support_leg
        - Copies target poses to initial poses for next step
        - All FSP frames use (x, y, theta) format in global frame
        - Order of operations is important to avoid frame corruption
        """
        # Update swing state based on next support leg
        self.left_is_swing = (self.next_support_leg == SupportLeg.RIGHT)

        # Previous swing foot becomes new support foot
        self.left_foot_fsp['initial'] = self.left_foot_fsp['target'].copy()
        self.right_foot_fsp['initial'] = self.right_foot_fsp['target'].copy()
        self.torso_fsp['initial'] = self.torso_fsp['target'].copy()
        
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Swapped feet - New state:")
            logging.debug(f"  Support foot: {'Right' if self.left_is_swing else 'Left'}")
            logging.debug(f"  Swing foot: {'Left' if self.left_is_swing else 'Right'}")
    
    def calculate_com_trajectory(self, t_time: float, x_com_2d: np.ndarray, initial_torso: np.ndarray, target_torso: np.ndarray, rotation_as_quaternion: bool = False) -> np.ndarray:
        """Calculate the COM trajectory in global frame using preview control output.

        This method generates a smooth center of mass (COM) trajectory by:
        1. Using preview control output for x-y position
        2. Maintaining constant COM height
        3. Interpolating torso orientation using SLERP between initial and target poses

        Parameters
        ----------
        t_time : float
            Current time within the step cycle [0, t_step] in seconds
        x_com_2d : np.ndarray, shape (2,)
            2D position [x, y] of the COM from preview controller in global frame

        Returns
        -------
        np.ndarray, shape (4, 4)
            Homogeneous transformation matrix representing the COM pose in global frame:
            - Translation: [x, y] from preview control, z from config
            - Rotation: SLERP interpolation between initial and target torso orientations

        Notes
        -----
        - All positions and orientations are in the global coordinate system
        - Uses h-function from footstep planner for smooth interpolation
        - Maintains constant COM height from preview controller config
        """
        norm_t_time = t_time / self.fsp.t_step
        initial_torso_quat = convert_rotation_to_quaternion(get_rotation_z(self.torso_fsp['initial'][2]))
        target_torso_quat = convert_rotation_to_quaternion(get_rotation_z(self.torso_fsp['target'][2]))
        
        h_func = self.fsp.calcHfunc(t_time, norm_t_time)
        torso_quat = h_func * initial_torso_quat + (1 - h_func) * target_torso_quat
        torso_quat = (torso_quat / np.linalg.norm(torso_quat)).ravel()

        if rotation_as_quaternion:
            return np.array([x_com_2d[0], x_com_2d[1], self.pc.com_height] + torso_quat.ravel().tolist())
        com_transforms = np.eye(4, dtype=np.float32)
        com_transforms[:3, :3] = convert_quaternion_to_rotation(torso_quat)
        com_transforms[:3, 3] = [x_com_2d[0], x_com_2d[1], self.pc.com_height]
        return com_transforms
    
    def calculate_swing_foot(self, t_time: float, swing_foot_initial: np.ndarray, swing_foot_target: np.ndarray, rotation_as_quaternion: bool = False) -> np.ndarray:
        """Calculate the swing foot trajectory in global frame during a step.

        This method generates a smooth swing foot trajectory by:
        1. Interpolating position between initial and target poses using h-function
        2. Interpolating orientation using quaternion SLERP
        3. Adding vertical motion using v-function for foot clearance

        Parameters
        ----------
        t_time : float
            Current time within the step cycle [0, t_step] in seconds

        Returns
        -------
        np.ndarray, shape (4, 4)
            Homogeneous transformation matrix for swing foot in global frame:
            - Translation: [x, y] interpolated, z follows step height curve
            - Rotation: SLERP between initial and target orientations

        Notes
        -----
        - All poses are in global coordinate system
        - Uses h-function for horizontal interpolation (position and orientation)
        - Uses v-function for vertical trajectory (bell-shaped curve)
        - Vertical motion: z = 0 -> step_height -> 0 during swing phase
        - Includes foot_offset for static height adjustment
        """
        norm_t_time = t_time / self.fsp.t_step
        h_phase = self.fsp.calcHfunc(t_time, norm_t_time)
        v_phase = self.fsp.calcVfunc(t_time, norm_t_time)

        initial_swing_quat = convert_rotation_to_quaternion(get_rotation_z(swing_foot_initial[2]))
        target_swing_quat = convert_rotation_to_quaternion(get_rotation_z(swing_foot_target[2]))
        swing_horizontal = h_phase * swing_foot_target[:2] + \
            (1 - h_phase) * swing_foot_initial[:2]

        swing_quat = h_phase * target_swing_quat + \
            (1 - h_phase) * initial_swing_quat
        swing_quat = (swing_quat / np.linalg.norm(swing_quat)).ravel()

        swing_vertical = self.step_height * v_phase + float(self.foot_offset[2])

        if rotation_as_quaternion:
            return np.array([swing_horizontal[0], swing_horizontal[1], swing_vertical] + swing_quat.ravel().tolist())
        swing_foot_pose = np.eye(4, dtype=np.float32)
        swing_foot_pose[:3, :3] = convert_quaternion_to_rotation(swing_quat)
        swing_foot_pose[:2, 3] = swing_horizontal.ravel()
        swing_foot_pose[2, 3] = swing_vertical
        return swing_foot_pose

    def calculate_support_foot(self, t_time: float, support_foot_initial: np.ndarray, support_foot_target: np.ndarray, rotation_as_quaternion: bool = False) -> np.ndarray:
        """Calculate the support foot pose in global frame during stance phase.

        This method computes the support foot trajectory, which should remain
        relatively stationary during the step. While horizontal interpolation
        is calculated for consistency, the foot should maintain ground contact.

        Parameters
        ----------
        t_time : float
            Current time within the step cycle [0, t_step] in seconds

        Returns
        -------
        np.ndarray, shape (4, 4)
            Homogeneous transformation matrix for support foot in global frame:
            - Translation: [x, y] interpolated, z fixed at foot_offset
            - Rotation: Fixed at target orientation

        Notes
        -----
        - All poses are in global coordinate system
        - Uses h-function for horizontal interpolation (position only)
        - Maintains constant height at foot_offset
        - No vertical motion to ensure ground contact
        - Orientation fixed to target to maintain stability
        """
        norm_t_time = t_time / self.fsp.t_step
        h_phase = self.fsp.calcHfunc(t_time, norm_t_time)
        
        support_horizontal = h_phase * support_foot_target[:2] + \
            (1 - h_phase) * support_foot_initial[:2]
        
        # The support foot should stay on the ground (z=0)
        support_vertical = float(self.foot_offset[2])  # Only apply foot offset if any
        rotation_z = get_rotation_z(support_foot_target[2])
        if rotation_as_quaternion:
            return np.array([support_horizontal[0], support_horizontal[1], support_vertical] + convert_rotation_to_quaternion(rotation_z).ravel().tolist())
        support_foot_pose = np.eye(4, dtype=np.float32)
        support_foot_pose[:3, :3] = rotation_z
        support_foot_pose[:2, 3] = support_horizontal.ravel()
        support_foot_pose[2, 3] = support_vertical
        return support_foot_pose

    def update(self, robot_kd: KinematicsDynamics) -> None:
        """
        Update the walking engine for the current time step.
        
        This method performs the main walking cycle:
        1. Updates the current robot state from kinematics
        2. Plans footsteps and ZMP trajectory in the support foot frame
        3. Transforms results back to the global frame
        4. Calculates COM and swing foot trajectories
        5. Updates robot joint angles using inverse kinematics
        
        Parameters
        ----------
        dt : float
            Time step in seconds
        robot_kd : KinematicsDynamics
            KinematicsDynamics instance for accessing and updating robot state
        """
        raise NotImplementedError("Update method is not implemented")
        # Use robot_kd to get positions and orientations without storing it
        
        # Calculate the target torso and feet positions at the beginning of a step
        if self.t_sim == 0:
            # NOTE: Current swing foot is basically the last foot that was swung, hence it is the foot that will be set as the support foot
            next_supp_leg = SupportLeg.LEFT if self.left_is_swing else SupportLeg.RIGHT

            self._get_current_robot_state(robot_kd, next_supp_leg)

            logging.info(f"Next support leg: {next_supp_leg} - First step: {self.first_step}")
            logging.info(f"Current swing foot position: {self.swing_foot_fsp['initial'][:3, 3]}, rotation: {convert_rotation_to_rpy(self.swing_foot_fsp['target'][:3, :3])}")
            logging.info(f"Current support foot position: {self.support_foot_fsp['initial'][:3, 3]}, rotation: {convert_rotation_to_rpy(self.support_foot_fsp['target'][:3, :3])}")
            logging.info(f"Current torso position: {self.torso_fsp['initial'][:3, 3]}, rotation: {convert_rotation_to_rpy(self.torso_fsp['target'][:3, :3])}")
            # Transform frames to support foot frame for footstep planning
            # Extract current support foot pose as the reference frame
            support_foot_pose = np.zeros((3, 1), dtype=np.float32)
            support_foot_pose[:2, 0] = self.support_foot_fsp['initial'][:2, 3]  # x, y position (in global frame)
            support_foot_pose[2, 0] = convert_rotation_to_rpy(self.support_foot_fsp['initial'][:3, :3])[2]  # yaw angle
            
            # Extract torso pose in global frame
            torso_pose = np.zeros((3, 1), dtype=np.float32)
            torso_pose[:2, 0] = self.torso_fsp['initial'][:2, 3]  # x, y position
            torso_pose[2, 0] = convert_rotation_to_rpy(self.torso_fsp['initial'][:3, :3])[2]  # yaw angle
            
            # Transform torso pose to support foot frame (support foot is at origin in its own frame)
            torso_in_support_frame = self.fsp.transform_from_frame(torso_pose, support_foot_pose)
            
            # Plan up to N steps in support foot frame
            # Support foot is at origin (0,0,0) in its own frame
            # Torso is expressed relative to support foot frame
            foot_pos_plan, torso_pos_plan, zmp_horizon_, _ = self.fsp.calculate(
                self.cmd_vel, np.zeros((3, 1), dtype=np.float32),  # Support foot is at origin in its own frame
                torso_in_support_frame, next_supp_leg, sway=self.first_step)
            
            # Transform results back to global frame
            # Transform the planned torso and swing foot positions back to global frame (x,y,yaw)
            # Use transform_to_frame to transform from support foot frame to global frame
            # In this case, support_foot_pose is the reference frame in global coordinates
            torso_plan_global = self.fsp.transform_to_frame(np.asarray(torso_pos_plan[1][1], dtype=np.float32).reshape(3, 1), self.global_frame[:3, 3])
            swing_foot_plan_global = self.fsp.transform_to_frame(np.asarray(foot_pos_plan[1][1], dtype=np.float32).reshape(3, 1), self.global_frame[:3, 3])

            # Update positions in the target transformation matrices
            self.torso_fsp['target'][:2, 3] = torso_plan_global[:2, 0]
            # Update only the yaw component while preserving roll and pitch
            current_torso_rpy = convert_rotation_to_rpy(self.torso_fsp['initial'][:3, :3])
            self.torso_fsp['target'][:3, :3] = get_rotation_rpy(roll=current_torso_rpy[0], pitch=current_torso_rpy[1], yaw=torso_plan_global[2, 0])
            
            self.swing_foot_fsp['target'][:2, 3] = swing_foot_plan_global[:2, 0]
            # Update only the yaw component while preserving roll and pitch
            current_swing_foot_rpy = convert_rotation_to_rpy(self.swing_foot_fsp['initial'][:3, :3])
            self.swing_foot_fsp['target'][:3, :3] = get_rotation_rpy(roll=current_swing_foot_rpy[0], pitch=current_swing_foot_rpy[1], yaw=swing_foot_plan_global[2, 0])
            
            logging.info(f"Target swing foot position: {self.swing_foot_fsp['target'][:3, 3]}, rotation: {convert_rotation_to_rpy(self.swing_foot_fsp['target'][:3, :3])}")
            logging.info(f"Target support foot position: {self.support_foot_fsp['target'][:3, 3]}, rotation: {convert_rotation_to_rpy(self.support_foot_fsp['target'][:3, :3])}")
            logging.info(f"Target torso position: {self.torso_fsp['target'][:3, 3]}, rotation: {convert_rotation_to_rpy(self.torso_fsp['target'][:3, :3])}")
            
            # Transform ZMP points to global frame
            global_zmp_horizon = []
            for zmp in zmp_horizon_:
                # Create a 3D pose from the 2D ZMP by adding a zero orientation
                zmp_pose = np.zeros((3, 1), dtype=np.float32)
                zmp_pose[0, 0] = zmp[0]
                zmp_pose[1, 0] = zmp[1]
                global_zmp = self.fsp.transform_to_frame(zmp_pose, self.global_frame[:3, 3:4])
                global_zmp_horizon.append([global_zmp[0, 0], global_zmp[1, 0]])
            
            # Update ZMP buffer with transformed points
            self.zmp_horizon = np.asarray(global_zmp_horizon, dtype=np.float32)
        
        # Compute preview controller
        self.state_x, zmp_x, _ = self.pc.update_state_err(self.state_x, self.zmp_horizon[:, 0])
        self.state_y, zmp_y, _ = self.pc.update_state_err(self.state_y, self.zmp_horizon[:, 1])

        # Update ZMP
        self.zmp_2d = np.array([float(zmp_x), float(zmp_y), 0]).reshape((3, 1))

        # Calculate COM trajectory
        self.torso_traj = self.calculate_com_trajectory(self.t_sim, [float(self.state_x[0][0]), float(self.state_y[0][0])])

        # Calculate swing foot trajectory
        # NOTE: 4D transformation matrix is used
        self.swing_foot_traj = self.calculate_swing_foot(self.t_sim)
        self.support_foot_traj = self.calculate_support_foot(self.t_sim)

        # Solve ik using current foot positions and torso position in global frame
        valid_ik = True
        
        # Set the torso transformation in the robot model
        robot_kd.set_transformation(TORSO_ID, self.torso_traj)
        
        # Option 1: Use the original Jacobian-based IK method
        use_jacobian_ik = False
        
        if use_jacobian_ik:
            # Original Jacobian-based IK method (from global frame)
            if self.left_is_swing:
                logging.info(f'Computing target pos from torso: {self.torso_traj[:3, 3]} to left swing: {self.swing_foot_traj[:3, 3]}')
                if not robot_kd.calc_inverse_kinematics_from_to(from_id=TORSO_ID, to=LEFT_FOOT_ID, tar_position=self.swing_foot_traj[:3, 3].reshape(3, 1), tar_rotation=self.swing_foot_traj[:3, :3]):
                    logger.error("Failed to calculate inverse kinematics for left foot")
                    valid_ik = False
                logging.info(f'Computing target pos from torso: {self.torso_traj[:3, 3]} to right support: {self.support_foot_traj[:3, 3]}')
                if not robot_kd.calc_inverse_kinematics_from_to(from_id=TORSO_ID, to=RIGHT_FOOT_ID, tar_position=self.support_foot_traj[:3, 3].reshape(3, 1), tar_rotation=self.support_foot_traj[:3, :3]):
                    logger.error("Failed to calculate inverse kinematics for right foot")
                    valid_ik = False
            else:
                logging.info(f'Computing target pos from torso: {self.torso_traj[:3, 3]} to right swing: {self.swing_foot_traj[:3, 3]}')
                if not robot_kd.calc_inverse_kinematics_from_to(from_id=TORSO_ID, to=RIGHT_FOOT_ID, tar_position=self.swing_foot_traj[:3, 3].reshape(3, 1), tar_rotation=self.swing_foot_traj[:3, :3]):
                    logger.error("Failed to calculate inverse kinematics for right foot")
                    valid_ik = False
                logging.info(f'Computing target pos from torso: {self.torso_traj[:3, 3]} to left support: {self.support_foot_traj[:3, 3]}')
                if not robot_kd.calc_inverse_kinematics_from_to(from_id=TORSO_ID, to=LEFT_FOOT_ID, tar_position=self.support_foot_traj[:3, 3].reshape(3, 1), tar_rotation=self.support_foot_traj[:3, :3]):
                    logger.error("Failed to calculate inverse kinematics for left foot")
                    valid_ik = False
        else:
            # Option 2: Use the new analytical IK method (transform to local base frame first)
            # Create inverse of torso transformation to transform from global to local base frame
            torso_inv = np.linalg.inv(self.torso_traj)
            
            # Transform swing foot and support foot from global to local base frame
            swing_foot_local = np.dot(torso_inv, self.swing_foot_traj)
            support_foot_local = np.dot(torso_inv, self.support_foot_traj)
            
            # Extract position and rotation from the local transformations
            swing_foot_pos = swing_foot_local[:3, 3]
            swing_foot_rot = convert_rotation_to_rpy(swing_foot_local[:3, :3])
            
            support_foot_pos = support_foot_local[:3, 3]
            support_foot_rot = convert_rotation_to_rpy(support_foot_local[:3, :3])
            
            # logging.info(f'Local swing foot position: {swing_foot_pos}, rotation: {swing_foot_rot}')
            # logging.info(f'Local support foot position: {support_foot_pos}, rotation: {support_foot_rot}')
            
            # Apply analytical IK for both legs
            if self.left_is_swing:
                # Left leg is swing foot
                if not robot_kd.calc_analytical_inverse_kinematics_left_leg(swing_foot_pos, swing_foot_rot):
                    logger.error("Failed to calculate analytical inverse kinematics for left swing foot")
                    valid_ik = False
                
                # Right leg is support foot
                if not robot_kd.calc_analytical_inverse_kinematics_right_leg(support_foot_pos, support_foot_rot):
                    logger.error("Failed to calculate analytical inverse kinematics for right support foot")
                    valid_ik = False
            else:
                # Right leg is swing foot
                if not robot_kd.calc_analytical_inverse_kinematics_right_leg(swing_foot_pos, swing_foot_rot):
                    logger.error("Failed to calculate analytical inverse kinematics for right swing foot")
                    valid_ik = False
                
                # Left leg is support foot
                if not robot_kd.calc_analytical_inverse_kinematics_left_leg(support_foot_pos, support_foot_rot):
                    logger.error("Failed to calculate analytical inverse kinematics for left support foot")
                    valid_ik = False
                    
            # Update forward kinematics after setting all joint angles
            robot_kd.calc_forward_kinematics(0)

        # Update simulation time
        self.t_sim += self.dt_sim
        
        # Buffer FIFO: Pop ZMP reference at each iteration
        self.zmp_horizon = self.zmp_horizon[1:]
        
        # Check if we've completed a step
        if self.t_sim > self.fsp.t_step:
            print("#### SWAPPING FOOT!!!")
            self.t_sim = 0
            self.steps_count += 1
            self.swap_foot()

        return valid_ik