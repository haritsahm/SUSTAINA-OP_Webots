import numpy as np
from typing import List
from omegaconf import DictConfig
import logging
from .foot_step_planner import FootStepPlanner, SupportLeg
from .preview_control import PreviewControl
from ..kinematics_dynamics.kinematics_dynamics import KinematicsDynamics
from .mathematics import convert_rotation_to_quaternion, convert_quaternion_to_rotation, get_rotation_z, get_rotation_rpy, convert_rotation_to_rpy

# Global constants for joint IDs
TORSO_ID = 0       # Base/torso joint ID
LEFT_FOOT_ID = 30  # Left foot joint ID
RIGHT_FOOT_ID = 31 # Right foot joint ID

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
        """
        Initialize the OpenZMPWalk engine.
        
        Parameters
        ----------
        config : DictConfig
            Configuration parameters for the walking engine containing:
            - walk: Walking parameters (step frequency, height, etc.)
            - preview_control: Preview controller parameters
            - foot_step_planner: Footstep planner parameters
        """
        self.config = config
        
        # Initialize the preview controller and footstep planner
        self.pc = PreviewControl(config)
        self.fsp = FootStepPlanner(config)
        
        # Walking parameters
        self.step_height = 0.0     # meters
        self.foot_separation = 0.0 # meters
        
        # Walking command
        self.cmd_vel = np.zeros((3, 1), dtype=np.float32)  # [x_speed, y_speed, a_speed]
        
        # State variables
        self.t_sim = 0.0     # Simulation time within a step
        self.dt_sim = 0.01   # Simulation time step
        self.left_is_swing = False   # True if left foot is swinging (right is support)
        self.first_step = True       # Flag for the first step
        self.steps_count = 1         # Counter for steps taken

        # Global reference frame (fixed at origin)
        # Identity homogeneous transformation matrix (4x4)
        self.global_frame = np.eye(4, dtype=np.float32)  # Fixed reference frame at origin
        
        # Frame containers with respect to global frame
        # Each container has current and target frames
        # Each frame is a 4x4 homogeneous transformation matrix containing both rotation and translation
        # [ R(3x3) t(3x1) ]
        # [   0      1    ]
        # where R is the rotation matrix and t is the translation vector
        self.torso = {
            'current': np.eye(4, dtype=np.float32),  # Current torso frame w.r.t global frame
            'target': np.eye(4, dtype=np.float32)    # Target torso frame w.r.t global frame
        }
        
        self.support_foot = {
            'current': np.eye(4, dtype=np.float32),  # Current support foot frame w.r.t global frame
            'target': np.eye(4, dtype=np.float32)    # Target support foot frame w.r.t global frame
        }
        
        self.swing_foot = {
            'current': np.eye(4, dtype=np.float32),  # Current swing foot frame w.r.t global frame
            'target': np.eye(4, dtype=np.float32)    # Target swing foot frame w.r.t global frame
        }
        
        # Intermediate calculation containers
        self.swing_foot_traj = np.eye(4, dtype=np.float32)  # x,y,theta,z
        self.supp_foot_traj = np.eye(4, dtype=np.float32)   # x,y,theta,z
        self.torso_traj = np.eye(4, dtype=np.float32)       # x,y,z,theta

        # ZMP and COM state variables
        self.zmp_horizon = np.zeros((0, 2), dtype=np.float32)  # ZMP reference buffer
        self.zmp_2d = np.zeros((3,1), dtype=np.float32)  # Current ZMP position
        
        # Foot offset for fine-tuning foot positions
        self.foot_offset = np.zeros((3,1), dtype=np.float32)
        
        # Initialize parameters from config
        self._load_parameters()
        
        # Initialize frames with default values
        self.reset_frames()
        
    def reset_frames(self, robot_kd: KinematicsDynamics = None):
        """
        Initialize or reset the frames based on robot kinematics and dynamics.
        
        This method initializes the torso, support foot, and swing foot frames
        using homogeneous transformation matrices. It always sets left_is_swing to True
        to ensure walking starts with left leg swinging first.
        
        This method is useful when the robot falls or needs to restart walking from its initial stance.
        
        Parameters
        ----------
        robot_kd : KinematicsDynamics, optional
            KinematicsDynamics instance for accessing robot state.
            If None, frames are initialized with default values.
        """
        # Always start with left leg swinging first
        self.left_is_swing = True
        
        # Reset step counter and simulation time
        self.t_sim = 0
        self.steps_count = 1

        # Initialize frames from robot kinematics
        # Get torso position and orientation
        torso_pos = robot_kd.get_position(TORSO_ID)
        torso_orient = robot_kd.get_rotation(TORSO_ID)
        # Update torso homogeneous transformation matrix
        self.torso['current'][:3, :3] = torso_orient  # Set rotation matrix (3x3)
        self.torso['current'][:3, 3] = torso_pos      # Set translation vector (3x1)
        
        # Get left and right foot positions and orientations
        left_foot_pos = robot_kd.get_position(LEFT_FOOT_ID)
        right_foot_pos = robot_kd.get_position(RIGHT_FOOT_ID)
        
        left_foot_orient = robot_kd.get_rotation(LEFT_FOOT_ID)
        right_foot_orient = robot_kd.get_rotation(RIGHT_FOOT_ID)
        
        # Create homogeneous transformation matrices for left and right feet
        left_foot_transform = np.eye(4, dtype=np.float32)
        left_foot_transform[:3, :3] = left_foot_orient
        left_foot_transform[:3, 3] = left_foot_pos
        
        right_foot_transform = np.eye(4, dtype=np.float32)
        right_foot_transform[:3, :3] = right_foot_orient
        right_foot_transform[:3, 3] = right_foot_pos
        
        # Since left_is_swing is always True when resetting, right foot is always support
        self.support_foot['current'] = right_foot_transform
        self.swing_foot['current'] = left_foot_transform

        # Set target positions to match current positions (no movement initially)
        self.torso['target'] = self.torso['current'].copy()
        self.support_foot['target'] = self.support_foot['current'].copy()
        self.swing_foot['target'] = self.swing_foot['current'].copy()

        # Initialize preview control state
        self.state_x = self.pc.init_state_err(pos=float(self.torso['current'][0]), e=0)
        self.state_y = self.pc.init_state_err(pos=float(self.torso['current'][1]), e=0)

    def _get_current_robot_state(self, robot_kd: KinematicsDynamics) -> None:
        """
        Get current robot state from kinematics and dynamics.
        
        Updates the current torso, support foot, and swing foot frames from the
        KinematicsDynamics instance without resetting other parameters like
        left_is_swing or step counters. This method extracts the position and
        orientation of each frame and updates the homogeneous transformation matrices.
        
        Parameters
        ----------
        robot_kd : KinematicsDynamics
            KinematicsDynamics instance for accessing robot state
        """
        # Get torso position and orientation
        torso_pos = robot_kd.get_position(TORSO_ID)
        torso_orient = robot_kd.get_rotation(TORSO_ID)
        
        # Update torso homogeneous transformation matrix
        self.torso['current'][:3, :3] = torso_orient  # Set rotation matrix (3x3)
        self.torso['current'][:3, 3] = torso_pos      # Set translation vector (3x1)
        
        # Get left and right foot positions and orientations
        left_foot_pos = robot_kd.get_position(LEFT_FOOT_ID)
        right_foot_pos = robot_kd.get_position(RIGHT_FOOT_ID)
        
        left_foot_orient = robot_kd.get_rotation(LEFT_FOOT_ID)
        right_foot_orient = robot_kd.get_rotation(RIGHT_FOOT_ID)
        
        # Create homogeneous transformation matrices for left and right feet
        left_foot_transform = np.eye(4, dtype=np.float32)
        left_foot_transform[:3, :3] = left_foot_orient
        left_foot_transform[:3, 3] = left_foot_pos
        
        right_foot_transform = np.eye(4, dtype=np.float32)
        right_foot_transform[:3, :3] = right_foot_orient
        right_foot_transform[:3, 3] = right_foot_pos
        
        # Update support and swing foot based on current state
        if self.left_is_swing:
            self.support_foot['current'] = right_foot_transform
            self.swing_foot['current'] = left_foot_transform
        else:
            self.support_foot['current'] = left_foot_transform
            self.swing_foot['current'] = right_foot_transform

    def _load_parameters(self):
        """
        Load parameters from configuration and initialize foot positions
        """
        # Load walking parameters from config
        if 'zmp_walk' in self.config:
            zmp_config = self.config.zmp_walk
            self.step_height = zmp_config.get('step_height', 0.04)  # Default step height

            # Load foot offset if available
            if 'foot_offset' in zmp_config:
                foot_offset = zmp_config.foot_offset
                self.foot_offset = np.array([
                    [foot_offset.get('x', 0.0)], 
                    [foot_offset.get('y', 0.0)], 
                    [foot_offset.get('z', 0.0)]
                ], dtype=np.float32)
        else:
            logger.warning("No ZMP walk configuration found, using default values")
            self.step_height = 0.04
            self.foot_offset = np.zeros((3,1), dtype=np.float32)

        # Set the time step from the preview controller
        self.dt_sim = self.pc.dt
        self.foot_separation = self.fsp.y_sep  # Use the value from footstep planner

        # Initialize foot positions based on foot separation
        # Left foot is positioned at y = foot_separation/2
        # Right foot is positioned at y = -foot_separation/2
        # Both feet start at x = 0, z = 0

        # Set initial positions for left and right feet as column vectors
        left_foot_init = np.array([[0.0], [self.foot_separation], [0.0]], dtype=np.float32)
        right_foot_init = np.array([[0.0], [-self.foot_separation], [0.0]], dtype=np.float32)

        # Set initial torso position (midpoint between feet)
        torso_init = np.zeros((3,1), dtype=np.float32)

        # Initialize frames with initial positions
        self.torso['current'][:3, 3] = torso_init
        self.torso['current'][:3, :3] = get_rotation_z(0.0)  # Initial yaw
        self.torso['target'] = self.torso['current'].copy()

        # Initialize support and swing foot positions based on which foot starts as support
        if self.left_is_swing:
            # Right foot is support, left foot is swing
            self.support_foot['current'][:3, 3] = right_foot_init
            self.support_foot['current'][:3, :3] = get_rotation_z(0.0)
            self.swing_foot['current'][:3, 3] = left_foot_init
            self.swing_foot['current'][:3, :3] = get_rotation_z(0.0)
        else:
            # Left foot is support, right foot is swing
            self.support_foot['current'][:3, 3] = left_foot_init
            self.support_foot['current'][:3, :3] = get_rotation_z(0.0)
            self.swing_foot['current'][:3, 3] = right_foot_init
            self.swing_foot['current'][:3, :3] = get_rotation_z(0.0)
        
        # Set target positions to match current positions (no movement initially)
        self.support_foot['target'] = self.support_foot['current'].copy()
        self.swing_foot['target'] = self.swing_foot['current'].copy()
    
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
    
    def swap_foot(self):
        """
        Switch the foot from left to right or vice versa
        """
        if not self.first_step:
            # Swap the support and swing feet
            temp = self.support_foot['current'].copy()
            
            if self.left_is_swing:
                # Left was swing, now becomes support
                self.left_is_swing = False
                # Update current positions
                self.support_foot['current'] = self.swing_foot['current'].copy()  # Left foot becomes support
                self.swing_foot['current'] = temp  # Right foot becomes swing
            else:
                # Right was swing, now becomes support
                self.left_is_swing = True
                # Update current positions
                self.support_foot['current'] = self.swing_foot['current'].copy()  # Right foot becomes support
                self.swing_foot['current'] = temp  # Left foot becomes swing
            
            # Update torso position
            self.torso['current'] = self.torso['target'].copy()
        
        self.first_step = False
    
    def calculate_com_trajectory(self, t_time: float, x_com_2d: List[float]) -> np.ndarray:
        """
        Calculate the COM trajectory based on the preview control output.
        
        This method computes the center of mass (COM) trajectory by interpolating
        between the current and target torso positions and orientations. It uses
        quaternion interpolation (SLERP) for smooth rotation transitions.
        
        Parameters
        ----------
        t_time : float
            Current time within the step (seconds)
        x_com_2d : List[float]
            2D position of the COM [x, y] from the preview controller
            
        Returns
        -------
        np.ndarray
            4x4 homogeneous transformation matrix for the COM
        """
        norm_t_time = t_time / self.fsp.t_step
        # NOTE: Using SLERP for smooth rotation interpolation between quaternions
        current_torso_quat = convert_rotation_to_quaternion(self.torso['current'][:3, :3])
        target_torso_quat = convert_rotation_to_quaternion(self.torso['target'][:3, :3])
        
        h_func = self.fsp.calcHfunc(t_time, norm_t_time)
        # Linear interpolation of quaternions
        torso_quat = h_func * current_torso_quat + (1 - h_func) * target_torso_quat
        # Normalize the quaternion to ensure unit length
        torso_quat = torso_quat / np.linalg.norm(torso_quat)

        com_translation = np.array([x_com_2d[0], x_com_2d[1], self.pc.com_height])
        com_rotation = convert_quaternion_to_rotation(torso_quat)
        com_transforms = np.eye(4, dtype=np.float32)
        com_transforms[:3, :3] = com_rotation
        com_transforms[:3, 3] = com_translation
        return com_transforms
    
    def calculate_swing_foot(self, t_time: float) -> np.ndarray:
        """
        Calculate the swing foot trajectory during a step.
        
        This method computes the swing foot trajectory by:
        1. Interpolating horizontally between current and target positions
        2. Generating a vertical trajectory with a bell-shaped curve for foot clearance
        3. Combining these into a homogeneous transformation matrix
        
        The horizontal interpolation uses the H-function from the footstep planner,
        while the vertical trajectory uses the V-function to create a smooth lifting motion.
        
        Parameters
        ----------
        t_time : float
            Current time within the step (seconds)
            
        Returns
        -------
        np.ndarray
            4x4 homogeneous transformation matrix for the swing foot
        """
        norm_t_time = t_time / self.fsp.t_step
        h_phase = self.fsp.calcHfunc(t_time, norm_t_time)
        
        if self.first_step:
            norm_t_time = 0
            h_phase = 0
        
        # Interpolate between current and target positions
        swing_horizontal = h_phase * self.swing_foot['target'][:2] + \
            (1 - h_phase) * self.swing_foot['current'][:2]
        
        v_func = self.fsp.calcVfunc(t_time, norm_t_time)
        swing_vertical = self.step_height * v_func + float(self.foot_offset[2])
        
        swing_foot_pose = np.eye(4, dtype=np.float32)
        swing_foot_pose[:3, 3] = np.array([swing_horizontal[0], swing_horizontal[1], swing_vertical])
        swing_foot_pose[:3, :3] = self.swing_foot['target'][:3, :3]
        return swing_foot_pose

    def update(self, dt: float, robot_kd: KinematicsDynamics) -> None:
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
        # Use robot_kd to get positions and orientations without storing it
        self._get_current_robot_state(robot_kd)
        
        # Calculate the target torso and feet positions at the beginning of a step
        if self.t_sim == 0:
            # NOTE: Current swing foot is basically the last foot that was swung, hence it is the foot that will be set as the support foot
            next_supp_leg = SupportLeg.LEFT if self.left_is_swing else SupportLeg.RIGHT
            
            # Transform frames to support foot frame for footstep planning
            # Extract current support foot pose as the reference frame
            support_foot_pose = np.zeros((3, 1), dtype=np.float32)
            support_foot_pose[:2, 0] = self.support_foot['current'][:2, 3]  # x, y position (in global frame)
            support_foot_pose[2, 0] = convert_rotation_to_rpy(self.support_foot['current'][:3, :3])[2]  # yaw angle
            
            # Extract torso pose in global frame
            torso_pose = np.zeros((3, 1), dtype=np.float32)
            torso_pose[:2, 0] = self.torso['current'][:2, 3]  # x, y position
            torso_pose[2, 0] = convert_rotation_to_rpy(self.torso['current'][:3, :3])[2]  # yaw angle
            
            # Transform torso pose to support foot frame (support foot is at origin in its own frame)
            torso_in_support_frame = self.fsp.transform_from_frame(torso_pose, support_foot_pose)
            
            # Plan up to N steps in support foot frame
            # Support foot is at origin (0,0,0) in its own frame
            # Torso is expressed relative to support foot frame
            foot_pos_plan, torso_pos_plan, zmp_horizon_, _ = self.fsp.calculate(
                self.cmd_vel, np.zeros((3, 1), dtype=np.float32),  # Support foot is at origin in its own frame
                torso_in_support_frame, next_supp_leg, sway=self.first_step)
            
            # Transform results back to global frame
            # Create target transformation matrices
            self.torso['target'] = self.torso['current'].copy()
            self.swing_foot['target'] = self.swing_foot['current'].copy()
            self.supp_foot_traj = self.support_foot['current'].copy()
            
            # Transform the planned torso and swing foot positions back to global frame (x,y,yaw)
            # Use transform_to_frame to transform from support foot frame to global frame
            # In this case, support_foot_pose is the reference frame in global coordinates
            torso_plan_global = self.fsp.transform_to_frame(np.asarray(torso_pos_plan[1][1:4], dtype=np.float32).reshape((3, 1)), self.global_frame[:3, 3])
            swing_foot_plan_global = self.fsp.transform_to_frame(np.asarray(foot_pos_plan[1][1:4], dtype=np.float32).reshape((3, 1)), self.global_frame[:3, 3])
            
            # Update positions in the target transformation matrices
            self.torso['target'][:2, 3] = torso_plan_global[:2, 0]
            # Update only the yaw component while preserving roll and pitch
            current_torso_rpy = convert_rotation_to_rpy(self.torso['current'][:3, :3])
            self.torso['target'][:3, :3] = get_rotation_rpy(roll=current_torso_rpy[0], pitch=current_torso_rpy[1], yaw=torso_plan_global[2, 0])
            
            self.swing_foot['target'][:2, 3] = swing_foot_plan_global[:2, 0]
            # Update only the yaw component while preserving roll and pitch
            current_swing_foot_rpy = convert_rotation_to_rpy(self.swing_foot['current'][:3, :3])
            self.swing_foot['target'][:3, :3] = get_rotation_rpy(roll=current_swing_foot_rpy[0], pitch=current_swing_foot_rpy[1], yaw=swing_foot_plan_global[2, 0])
            
            # Transform ZMP points to global frame
            global_zmp_horizon = []
            for zmp in zmp_horizon_:
                # Create a 3D pose from the 2D ZMP by adding a zero orientation
                zmp_pose = np.zeros((3, 1), dtype=np.float32)
                zmp_pose[0, 0] = zmp[0]
                zmp_pose[1, 0] = zmp[1]
                global_zmp = self.fsp.transform_to_frame(zmp_pose, self.global_frame[:3, 3])
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

        # Solve ik using current foot positions and torso position in global frame
        robot_kd.set_transformation(TORSO_ID, self.torso_traj)
        if self.left_is_swing:
            robot_kd.calc_inverse_kinematics_from_to(from_id=TORSO_ID, to_id=LEFT_FOOT_ID, tar_position=self.swing_foot_traj[:3, 3], tar_rotation=self.swing_foot_traj[:3, :3])
            robot_kd.calc_inverse_kinematics_from_to(from_id=TORSO_ID, to_id=RIGHT_FOOT_ID, tar_position=self.support_foot['current'][:3, 3], tar_rotation=self.support_foot['current'][:3, :3])
        else:
            robot_kd.calc_inverse_kinematics_from_to(from_id=TORSO_ID, to_id=RIGHT_FOOT_ID, tar_position=self.swing_foot_traj[:3, 3], tar_rotation=self.swing_foot_traj[:3, :3])
            robot_kd.calc_inverse_kinematics_from_to(from_id=TORSO_ID, to_id=LEFT_FOOT_ID, tar_position=self.support_foot['current'][:3, 3], tar_rotation=self.support_foot['current'][:3, :3])

        # Update simulation time
        self.t_sim += self.dt_sim
        
        # Buffer FIFO: Pop ZMP reference at each iteration
        self.zmp_horizon = self.zmp_horizon[1:]
        
        # Check if we've completed a step
        if self.t_sim > self.fsp.t_step:
            self.t_sim = 0
            self.steps_count += 1
            self.swap_foot()
