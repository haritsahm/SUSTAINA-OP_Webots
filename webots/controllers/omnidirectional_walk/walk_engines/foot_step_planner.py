import numpy as np
from typing import Tuple, List, Union, Dict
from omegaconf import DictConfig
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class SupportLeg(Enum):
    """Enum representing the support leg phase during walking.
    
    This enum is used to represent which leg is currently supporting the robot's weight:
    - LEFT (0): Left leg is the support leg, right leg is swinging
    - RIGHT (1): Right leg is the support leg, left leg is swinging
    - BOTH (2): Both legs are supporting (double support phase)
    
    Using integer values for more efficient storage and comparison.
    """
    LEFT = 0
    RIGHT = 1
    BOTH = 2


class FootStepPlanner():
    """Foot step planner for N consecutive steps in omnidirectional walking.
    
    This class generates footstep plans and ZMP reference trajectories for
    omnidirectional walking based on velocity commands. It implements the approach
    described in Marcos Maximo's research on ZMP-based walking for humanoid robots.
    
    The planner calculates N consecutive steps and generates the corresponding ZMP
    trajectory that can be used by the Preview Controller for stable walking.

    Parameters
    ----------
    config : DictConfig
        Configuration parameters from config.yaml
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the FootStepPlanner with configuration parameters.
        
        Parameters
        ----------
        config : DictConfig
            Configuration parameters from config.yaml
        """
        # Extract configuration parameters
        if 'foot_step_planner' not in config:
            logger.warning("Missing 'foot_step_planner' section in configuration, using defaults")
            planner_config = {}
        else:
            planner_config = config.foot_step_planner
            
        # Time Parameters
        self.t_step = planner_config.get('t_step', 0.25)  # Time for a single step (seconds)
        self.dsp_ratio = planner_config.get('dsp_ratio', 0.15)  # Double support phase ratio
        self.dt = planner_config.get('dt', 0.01)  # Sampling time (seconds)
        
        # Phase parameters
        self.ssp_phase = planner_config.get('ssp_phase', 0.2)  # Single support phase ratio
        self.dsp_phase = 1.0 - self.ssp_phase  # Double support phase ratio
        
        # Derived time parameters
        self.t_dsp = self.dsp_ratio * self.t_step  # Double support phase duration
        self.t_dsp_1 = self.t_dsp / 2  # First half of double support phase
        self.t_dsp_2 = self.t_dsp / 2  # Second half of double support phase
        
        # Phase transition times
        self.t_begin = self.t_dsp / 2  # Time when single support phase begins
        self.t_end = self.t_step - (self.t_dsp / 2)  # Time when single support phase ends
        self.norm_t_begin = self.t_begin / self.t_step  # Normalized begin time [0,1]
        self.norm_t_end = self.t_end / self.t_step  # Normalized end time [0,1]

        # Walking Parameters
        self.n_steps = planner_config.get('n_steps', 4)  # Number of steps to plan ahead
        if self.n_steps < 3:
            logger.warning("Number of steps must be >= 3, setting to 3")
            self.n_steps = 3
            
        self.y_sep = planner_config.get('foot_separation', 0.03525)  # Half hip width (meters)

        # Maximum stride parameters
        self.max_stride_x = planner_config.get('max_stride_x', 0.05)  # Max forward step (meters)
        self.max_stride_y = planner_config.get('max_stride_y', 0.03)  # Max lateral step (meters)
        self.max_stride_th = planner_config.get('max_stride_th', 0.2)  # Max rotation step (radians)

    def limit_velocity(self, vel_cmd: Union[List[float], Tuple[float, float, float], np.ndarray]) -> np.ndarray:
        """Limit velocity commands to maximum allowed values.
        
        This ensures that the commanded velocities don't exceed the robot's physical capabilities,
        which helps maintain stability during walking.
        
        Parameters
        ----------
        vel_cmd : Union[List[float], Tuple[float, float, float], np.ndarray]
            Input velocity command (x, y, theta)
            
        Returns
        -------
        np.ndarray
            Limited velocity command
        """
        # Convert to numpy array if not already
        vel_cmd = np.asarray(vel_cmd, dtype=np.float32)
        
        # Extract components
        cmd_x, cmd_y, cmd_a = vel_cmd
        
        # Calculate maximum velocities based on stride limits and step time
        max_vel_x = self.max_stride_x / self.t_step
        max_vel_y = self.max_stride_y / self.t_step
        max_vel_a = self.max_stride_th / self.t_step
        
        # Limit linear velocities
        cmd_x = np.clip(cmd_x, -max_vel_x, max_vel_x)
        cmd_y = np.clip(cmd_y, -max_vel_y, max_vel_y)
        
        # Limit angular velocity
        cmd_a = np.clip(cmd_a, -max_vel_a, max_vel_a)
        
        return np.array([cmd_x, cmd_y, cmd_a])
    
    def mod_angle(self, a: float) -> float:
        """Mod angle to keep in [-pi, pi]

        Parameters
        ----------
        a : float
            Input angle (rad)

        Returns
        -------
        float
            Mod/clamped angle in [-pi, pi]
        """

        a = a % (2 * np.pi)
        if (a >= np.pi):
            a = a - 2 * np.pi
        return a

    def calcHfunc(self, t_time, norm_t):
        """Horizontal Trajectory function

        Reference: Maximo, Marcos - Omnidirectional ZMP-Based Walking for Humanoid
        Section 3.3 - CoM Trajectory Using 3D-LIPM

        Calculate the trajectory using C1-continuous spline interpolation eq. (3.36)

        Parameters
        ----------
        t_time : float
            The time since the beginning of the step
        norm_t : float
            Normalized time since the beginning of the step [0, 1]

        Returns
        -------
        float
            Value of interpolation at time t_time
        """
        h_func = 0
        if norm_t < self.norm_t_begin:
            h_func = 0
        elif norm_t >= self.norm_t_begin and norm_t < self.norm_t_end:
            h_func = 0.5 * \
                (1 - np.cos(np.pi * ((t_time - self.t_begin) / (self.t_end - self.t_begin))))
        elif norm_t >= self.norm_t_end:
            h_func = 1
        return h_func

    def calcVfunc(self, t_time, norm_t):
        """Vertical Trajectory function

        Reference: Maximo, Marcos - Omnidirectional ZMP-Based Walking for Humanoid
        Section 3.3 - CoM Trajectory Using 3D-LIPM

        Calculate the trajectory using C1-continuous spline interpolation eq. (3.39)

        Parameters
        ----------
        t_time : float
            The time since the beginning of the step
        norm_t : float
            Normalized time since the beginning of the step [0, 1]

        Returns
        -------
        float
            Value of interpolation at time t
        """

        v_func = 0
        if norm_t < self.norm_t_begin or norm_t >= self.norm_t_end:
            v_func = 0
        elif norm_t >= self.norm_t_begin and norm_t < self.norm_t_end:
            v_func = 0.5 * (1 - np.cos(2 * np.pi * ((norm_t -
                            self.norm_t_begin) / (self.norm_t_end - self.norm_t_begin))))
        return v_func

    def transform_to_frame(self, local_pose, reference_frame):
        """Transform a pose from local coordinates to reference frame coordinates.
        
        This function applies a coordinate transformation to express a pose given in
        local coordinates in terms of the reference frame coordinates.
        
        Mathematically, if we have:
        - A pose P_local = [x, y, θ] in local coordinates
        - A reference frame F = [x_f, y_f, θ_f]
        
        Then this function computes P_ref = [x', y', θ'] where:
        - x' = x_f + x*cos(θ_f) - y*sin(θ_f)
        - y' = y_f + x*sin(θ_f) + y*cos(θ_f)
        - θ' = θ_f + θ
        
        Parameters
        ----------
        local_pose : np.ndarray
            Pose in local coordinates as [x, y, θ]
        reference_frame : np.ndarray
            Reference frame as [x, y, θ]
        
        Returns
        -------
        np.ndarray
            The transformed pose in reference frame coordinates
        """

        local_pose = np.asarray(local_pose, dtype=np.float32).reshape((3, 1))
        reference_frame = np.asarray(reference_frame, dtype=np.float32).reshape((3, 1))

        assert local_pose.shape == (3, 1), 'Shape must be (3,1)'
        assert reference_frame.shape == (3, 1), 'Shape must be (3,1)'

        ca = np.cos(reference_frame[2])
        sa = np.sin(reference_frame[2])

        return np.array([reference_frame[0] + ca * local_pose[0] - sa * local_pose[1],
                        reference_frame[1] + sa * local_pose[0] + ca * local_pose[1],
                        reference_frame[2] + local_pose[2]])
        
    # Alias for backward compatibility
    pose_global2d = transform_to_frame

    def transform_from_frame(self, global_pose, reference_frame):
        """Transform a pose from reference frame coordinates to local coordinates.
        
        This function applies the inverse coordinate transformation to express a pose
        given in reference frame coordinates in terms of local coordinates.
        
        Mathematically, if we have:
        - A pose P_ref = [x, y, θ] in reference frame coordinates
        - A reference frame F = [x_f, y_f, θ_f]
        
        Then this function computes P_local = [x', y', θ'] where:
        - x' = (x - x_f)*cos(θ_f) + (y - y_f)*sin(θ_f)
        - y' = -(x - x_f)*sin(θ_f) + (y - y_f)*cos(θ_f)
        - θ' = θ - θ_f
        
        Parameters
        ----------
        global_pose : np.ndarray
            Pose in reference frame coordinates as [x, y, θ]
        reference_frame : np.ndarray
            Reference frame as [x, y, θ]
        
        Returns
        -------
        np.ndarray
            The transformed pose in local coordinates
        """

        global_pose = np.asarray(global_pose, dtype=np.float32).reshape((3, 1))
        reference_frame = np.asarray(reference_frame, dtype=np.float32).reshape((3, 1))

        assert global_pose.shape == (3, 1), 'Shape must be (3,1)'
        assert reference_frame.shape == (3, 1), 'Shape must be (3,1)'

        ca = np.cos(reference_frame[2])
        sa = np.sin(reference_frame[2])

        px = global_pose[0] - reference_frame[0]
        py = global_pose[1] - reference_frame[1]
        pa = global_pose[2] - reference_frame[2]
        return np.array([ca * px + sa * py, -sa * px + ca * py, self.mod_angle(pa)])
        
    # Alias for backward compatibility
    pose_relative2d = transform_from_frame

    def calcNextPose(self, vel_cmd, current_pose):
        """Calculate the next torso pose from the current pose and velocity command.

        Reference: Maximo, Marcos - Omnidirectional ZMP-Based Walking for Humanoid
        Section 3.2 - Selecting Next Torso and Swing Foot Poses

        This method implements the closed-form solution from equation (3.8) in the thesis,
        which is derived by integrating the nonlinear differential equations (3.5) over 
        a step duration T. This provides an exact solution for the next torso pose given
        the current pose and velocity commands.

        Parameters
        ----------
        vel_cmd : Union[List[float], Tuple[float, float, float], np.ndarray]
            Velocity command as [v_x, v_y, omega] where:
            - v_x: Forward velocity (m/s)
            - v_y: Lateral velocity (m/s)
            - omega: Angular velocity (rad/s)
        current_pose : np.ndarray
            Current torso pose as a 3x1 array [x, y, theta] where:
            - x, y: Position coordinates in the reference frame (m)
            - theta: Orientation angle (rad)

        Returns
        -------
        np.ndarray
            The next torso pose after a step duration T as a 3x1 array [x', y', theta']
        """
        # Extract velocity components
        v_x, v_y, omega = vel_cmd
        
        # Create a copy of the current pose to avoid modifying the input
        next_pose = current_pose.copy()
        
        # Handle the special case of straight-line motion (no rotation)
        if abs(omega) < 1e-6:  # Near-zero angular velocity
            # For straight-line motion, use direct integration
            current_theta = float(current_pose[2])
            cos_theta = np.cos(current_theta)
            sin_theta = np.sin(current_theta)
            
            # Update position based on forward and lateral velocities
            next_pose[0] += self.t_step * (v_x * cos_theta - v_y * sin_theta)
            next_pose[1] += self.t_step * (v_x * sin_theta + v_y * cos_theta)
            # No change in orientation for straight-line motion
        else:
            # For curved motion, use the closed-form solution from equation (3.8)
            # Calculate normalized velocities (ratio of linear to angular velocity)
            v_x_norm = v_x / omega
            v_y_norm = v_y / omega
            
            # Calculate half of the total angular change during the step
            half_delta_theta = omega * self.t_step / 2
            
            # Calculate the midpoint angle (current angle + half of the angular change)
            mid_angle = float(current_pose[2]) + half_delta_theta
            
            # Calculate trigonometric terms for the closed-form solution
            sin_half_delta = np.sin(half_delta_theta)
            cos_mid_angle = np.cos(mid_angle)
            sin_mid_angle = np.sin(mid_angle)
            
            # Update position using the closed-form solution
            next_pose[0] += 2 * (v_x_norm * sin_half_delta * cos_mid_angle - 
                                v_y_norm * sin_half_delta * sin_mid_angle)
            next_pose[1] += 2 * (v_x_norm * sin_half_delta * sin_mid_angle + 
                                v_y_norm * sin_half_delta * cos_mid_angle)
            
            # Update orientation by adding the total angular change
            next_pose[2] += omega * self.t_step
            
        return next_pose

    def calcTorsoFoot(self, vel_cmd: np.ndarray, current_torso: np.ndarray, left_is_swing: int):
        """Calculate the target torso and swing foot positions.

        Reference: Maximo, Marcos - Omnidirectional ZMP-Based Walking for Humanoid
        Section 3.2 - Selecting Next Torso and Swing Foot Poses

        This method calculates the next torso pose T[k+1] and next swing foot pose A[k+1]
        from the current torso pose T[k] and current support foot pose S[k].
        
        IMPORTANT: All poses (T[k], T[k+1], etc.) in this method are defined with respect 
        to the support foot frame S[k]. The method works entirely in the support foot frame,
        avoiding unnecessary transformations to and from the global frame.

        Parameters
        ----------
        vel_cmd : Union[List[float], Tuple[float, float, float], np.ndarray]
            Velocity command as [v_x, v_y, omega] in support foot frame
        current_torso : Union[List[float], Tuple[float, float, float], np.ndarray]
            Current torso pose T[k] as [x, y, theta] in support foot frame
        left_is_swing : int
            Flag indicating if left leg is swinging (1) or right leg is swinging (0)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            target_torso_pos: Next torso pose T[k+1] as [x, y, theta] in support foot frame
            target_swing_pos: Next swing foot pose A[k+1] as [x, y, theta] in support foot frame
        """
        if isinstance(current_torso, list) or isinstance(current_torso, tuple):
            current_torso = np.asarray(current_torso, dtype=np.float32).reshape(3, 1)

        if isinstance(vel_cmd, list) or isinstance(vel_cmd, tuple):
            vel_cmd = np.asarray(vel_cmd)

        # Extract velocity components
        v_x, v_y, omega = vel_cmd
        
        # Store initial positions
        T_k = current_torso   # Current torso pose in support foot frame

        # Step 1: Calculate the next torso pose T[k+1] in support foot frame
        # Using the closed-form solution from the thesis (eq. 3.8)
        T_k1 = self.calcNextPose(vel_cmd, T_k)

        # Step 2: Determine if the leg configuration is open or closed
        # Check lateral open/close configuration using eq. (3.11)
        o_y = 1 if ((v_y >= 0 and left_is_swing) or
                    (v_y < 0 and not left_is_swing)) else 0

        # Check rotational open/close configuration using eq. (3.12)
        o_z = 1 if ((omega >= 0 and left_is_swing) or
                    (omega < 0 and not left_is_swing)) else 0

        # Step 3: Apply safety constraints to ensure stability
        # Step 3a: Ensure minimum y-separation between torso and support foot (eq. 3.10)
        if np.abs(float(T_k1[1])) < self.y_sep:
            # Adjust the y-coordinate to maintain minimum separation
            if float(T_k1[1]) >= 0:
                T_k1[1] = self.y_sep
            else:
                T_k1[1] = -self.y_sep

        # Step 3b: Apply angular safety constraint
        # Limit the torso orientation so the support foot never points inward
        angle_diff = float(T_k1[2])
        if np.abs(angle_diff) > 0:
            # Adjust orientation based on which leg is supporting
            if left_is_swing and angle_diff < 0:  # Right support
                T_k1[2] += np.abs(angle_diff)
            elif not left_is_swing and angle_diff > 0:  # Left support
                T_k1[2] -= np.abs(angle_diff)

        # Step 4: Calculate the next torso pose T[k+2] from the corrected T[k+1]
        # This is needed for swing foot placement calculations
        T_k2 = self.calcNextPose(vel_cmd, T_k1)

        # Step 5: Calculate the swing foot pose in reference frame P
        # Step 5a: Select reference frame P based on eq. (3.13)
        if o_z:  # Open configuration in rotation
            P = T_k2  # P = T[k+1+o_z] = T[k+2] in support foot frame
            # Calculate T_k1 relative to P (T_k2)
            T_k1_rel_P = self.pose_relative2d(T_k1, P)
            # T_k2 relative to itself is zero
            T_k2_rel_P = np.zeros((3, 1))
        else:  # Closed configuration in rotation
            P = T_k1  # P = T[k+1+o_z] = T[k+1] in support foot frame
            # T_k1 relative to itself is zero
            T_k1_rel_P = np.zeros((3, 1))
            # Calculate T_k2 relative to P (T_k1)
            T_k2_rel_P = self.pose_relative2d(T_k2, P)
            
        # Step 5b: Initialize next swing foot pose S[k+1] in P frame
        S_k1_rel_P = np.zeros((3, 1))
        
        # Step 5c: Set swing foot orientation using eq. (3.14)
        # This makes the axes of the swing foot the same as reference frame P
        S_k1_rel_P[2] = 0.0
        
        # Step 5d: Calculate lateral position (Y) of swing foot using eq. (3.15)
        # Position depends on open/closed configuration in lateral direction
        if o_y:  # Open configuration in lateral direction
            # Use T[k+2] with appropriate hip offset
            S_k1_rel_P[1] = T_k2_rel_P[1] + (np.power((-1), 1 - left_is_swing) * self.y_sep)
        else:  # Closed configuration in lateral direction
            # Use T[k+1] with appropriate hip offset
            S_k1_rel_P[1] = T_k1_rel_P[1] + (np.power((-1), 1 - left_is_swing) * self.y_sep)

        # Step 5e: Calculate forward position (X) of swing foot using eq. (3.20-3.22)
        # Handle the case where omega is zero or very close to zero
        # to avoid division by zero warnings
        epsilon = 1e-10  # Small threshold to check if omega is effectively zero
        
        if abs(omega) < epsilon:
            # For straight-line motion (no rotation), use simplified equations
            # The swing foot moves in a straight line
            diff_x = v_x * self.t_step
            # No normalization needed when omega is zero
            v_x_norm = 0
            v_y_norm = 0
            sin_half_delta = 0
            cos_half_delta = 1
        else:
            # Normalize velocity components by angular velocity for simplified calculations
            v_x_norm = v_x / omega
            v_y_norm = v_y / omega
            
            # Calculate half of the total angular change during the step
            half_delta_theta = omega * self.t_step / 2

            # Precompute sine and cosine for efficiency
            sin_half_delta = np.sin(half_delta_theta)
            cos_half_delta = np.cos(half_delta_theta)

        # Direct implementation of equation (3.20) from the thesis
        if o_z:  # Open configuration in rotation
            diff_x = 2 * v_x_norm * sin_half_delta * cos_half_delta + \
                2 * v_y_norm * sin_half_delta * sin_half_delta
        else:  # Closed configuration in rotation
            diff_x = 2 * v_x_norm * sin_half_delta * cos_half_delta - \
                2 * v_y_norm * sin_half_delta * sin_half_delta

        # Set forward position using equation (3.22) from the thesis
        # Places swing foot halfway between current and next torso positions
        S_k1_rel_P[0] = T_k1_rel_P[0] + (diff_x / 2)
        
        # Step 6: Transform the swing foot pose from P frame to support foot frame
        S_k1_rel_S_k = self.pose_global2d(S_k1_rel_P, P)
        S_k1_rel_S_k[2] = P[2]  # Set the orientation to match the reference frame

        return T_k1, S_k1_rel_S_k

    def compute_zmp_trajectory(self, zmp_t: float, init_torso_2d: np.ndarray, 
                             target_torso_2d: np.ndarray, init_supp_2d: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """Compute the ZMP trajectory with smoother transitions.

        Reference: Yi, Seung-Joon - Whole-Body Balancing Walk Controller
        for Position Controlled Humanoid Robots
        Section 3.1 - Footstep generation controller

        Compute the ZMP trajectory from the initial torso position
        to the target torso position through the support foot, using
        smoother transitions between phases for better stability.
        
        Note on smooth transitions:
        The original implementation used linear transitions between phases (m_dsp1 * t_time),
        which could cause jerky movements and instability at transition points. This enhanced
        version uses a cosine-based sigmoid function (0.5 * (1 - cos(π*t))) to create smooth
        S-shaped transitions that have continuous first derivatives at the boundaries.
        This results in more natural and stable robot movements by avoiding sudden
        accelerations that can excite the robot's dynamics in undesirable ways.

        Parameters
        ----------
        zmp_t : float
            Current ZMP time
        init_torso_2d : np.ndarray
            Initial torso position (x,y)
        target_torso_2d : np.ndarray
            Target torso position (x,y)
        init_supp_2d : np.ndarray
            Initial support leg position (x,y)

        Returns
        -------
        zmp_pos: List[np.ndarray]
            List of computed ZMP trajectory points
        timer_count: List[float]
            List of ZMP timer counts
        """
        # Ensure inputs are numpy arrays
        init_torso_2d = np.asarray(init_torso_2d, dtype=np.float32)
        target_torso_2d = np.asarray(target_torso_2d, dtype=np.float32)
        init_supp_2d = np.asarray(init_supp_2d, dtype=np.float32)

        # Initialize containers
        t_time = 0.0
        zmp_pos = []
        timer_count = []
        
        # Calculate slopes for transitions
        # m_dsp1 = (init_supp_2d - init_torso_2d) / self.t_dsp_1
        # m_dsp2 = (target_torso_2d - init_supp_2d) / self.t_dsp_2

        # Generate ZMP trajectory points
        while t_time < self.t_step:
            # First phase: transition from initial torso to support foot
            if t_time < self.t_begin:
                # Apply smooth transition using a sigmoid function for better stability
                progress = t_time / self.t_begin  # Normalized time [0,1]
                smooth_factor = 0.5 * (1 - np.cos(np.pi * progress))  # Smooth transition factor [0,1]
                x_zmp_2d = init_torso_2d + (init_supp_2d - init_torso_2d) * smooth_factor
                
            # Second phase: ZMP stays at support foot
            elif t_time >= self.t_begin and t_time < self.t_end:
                x_zmp_2d = init_supp_2d
                
            # Third phase: transition from support foot to target torso
            else:  # t_time >= self.t_end
                # Apply smooth transition using a sigmoid function for better stability
                progress = (t_time - self.t_end) / (self.t_step - self.t_end)  # Normalized time [0,1]
                smooth_factor = 0.5 * (1 - np.cos(np.pi * progress))  # Smooth transition factor [0,1]
                x_zmp_2d = init_supp_2d + (target_torso_2d - init_supp_2d) * smooth_factor
            
            # Store the ZMP position and time
            timer_count.append(zmp_t)
            zmp_pos.append(x_zmp_2d.ravel())
            
            # Increment time
            t_time += self.dt
            zmp_t += self.dt

        return zmp_pos, timer_count
    
    def get_zmp_reference_for_preview_control(self) -> np.ndarray:
        """Extract the ZMP reference trajectory in a format suitable for the PreviewControl class.
        
        This method formats the ZMP trajectory data to be compatible with the
        PreviewControl class, which expects a numpy array with shape (n_steps, 2)
        where each row contains [zmp_x, zmp_y] coordinates.
        
        Returns
        -------
        np.ndarray
            ZMP reference trajectory with shape (n_steps, 2) where each row is [zmp_x, zmp_y]
        """
        if not hasattr(self, 'zmp_pos') or not self.zmp_pos:
            logger.warning("No ZMP trajectory has been generated yet. Call calculate() first.")
            return np.array([])
            
        # Convert the list of ZMP positions to a numpy array
        # Each ZMP position is a 2D point [x, y]
        zmp_ref = np.array(self.zmp_pos)
        
        # Ensure the shape is correct (n_steps, 2)
        if zmp_ref.ndim == 1:
            # If we have a flat array, reshape it
            zmp_ref = zmp_ref.reshape(-1, 2)
        elif zmp_ref.shape[1] > 2:
            # If we have more than 2 columns, take only the first 2 (x,y)
            zmp_ref = zmp_ref[:, :2]
            
        return zmp_ref

    def calculate(self, vel_cmd: Union[List[float], Tuple[float, float, float], np.ndarray],
                current_swing: Union[List[float], Tuple[float, float, float], np.ndarray],
                current_torso: Union[List[float], Tuple[float, float, float], np.ndarray],
                next_support_leg: SupportLeg, sway: bool = True) -> Tuple[List[Tuple], List[Tuple], List[np.ndarray], List[float]]:
        """Calculate N consecutive steps from the given velocity commands.

        This method generates a sequence of footsteps, torso positions, and ZMP trajectory
        based on the current state and velocity commands. It handles the alternating
        support leg pattern and ensures smooth transitions between steps.
        
        Parameters
        ----------
        vel_cmd : Union[List[float], Tuple[float, float, float], np.ndarray]
            Velocity command input as [x_vel, y_vel, angular_vel]
        current_swing : Union[List[float], Tuple[float, float, float], np.ndarray]
            Current swing foot position as [x, y, theta]
        current_torso : Union[List[float], Tuple[float, float, float], np.ndarray]
            Current torso position as [x, y, theta]
        next_support_leg : str
            Next support leg, either 'left' or 'right'
        sway : bool, optional
            Whether to include initial sway in the ZMP trajectory, by default True
            
        Returns
        -------
        foot_step: List[Tuple]
            List of tuples containing (time, foot_position, leg_stance) for each foot step
        torso_pos: List[Tuple]
            List of tuples containing (time, torso_position) for each torso position
        zmp_pos: List[np.ndarray]
            List of ZMP trajectory positions as (x, y) numpy arrays
        timer_count: List[float]
            List of ZMP timer counts
            
        Raises
        ------
        ValueError
            If next_support_leg is not 'left' or 'right'
        """

        # Validate inputs
        if next_support_leg not in [SupportLeg.LEFT, SupportLeg.RIGHT]:
            raise ValueError("next_support_leg must be SupportLeg.LEFT or SupportLeg.RIGHT for planning")
            
        # Apply velocity limits for safety
        vel_cmd = self.limit_velocity(vel_cmd)
        
        # Initialize time and containers
        time = 0.0
        torso_pos = []
        foot_step = []
        zmp_pos = []
        zmp_t = 0
        timer_count = []
        
        # Set initial swing leg based on next support leg
        left_is_swing = 1 if next_support_leg == SupportLeg.RIGHT else 0

        # Add the initial state with both feet on the ground
        foot_step.append((time, current_swing, SupportLeg.BOTH))
        torso_pos.append((time, current_torso))
        
        # Move to the next time step
        time += self.t_step

        # Add initial ZMP reference to reduce initial error if sway is enabled
        if sway:
            # Generate initial ZMP points at the current torso position
            for _ in range(int(self.t_step // self.dt)):
                zmp_pos.append(np.asarray(current_torso[:2], dtype=np.float32))
                zmp_t += self.dt
                timer_count.append(zmp_t)

        # Compute the first step based on the next support leg
        if next_support_leg == SupportLeg.RIGHT:
            # Calculate target positions for torso and swing foot
            target_torso_pos, target_swing_pos = self.calcTorsoFoot(
                vel_cmd, current_torso, left_is_swing)

            # Switch support leg for the next step
            next_support_leg = SupportLeg.LEFT
            left_is_swing = 0

        elif next_support_leg == SupportLeg.LEFT:
            # Left leg is support, right leg is swinging
            # Calculate target positions for torso and swing foot
            target_torso_pos, target_swing_pos = self.calcTorsoFoot(
                vel_cmd, current_torso, left_is_swing)

            # Switch support leg for the next step
            next_support_leg = SupportLeg.RIGHT
            left_is_swing = 1

        # Compute ZMP trajectory for the first step
        temp_zmp, temp_tzmp = self.compute_zmp_trajectory(
            zmp_t, current_torso[:2], target_torso_pos[:2], current_swing[:2])

        zmp_pos.extend(temp_zmp)
        timer_count.extend(temp_tzmp)
        zmp_t = timer_count[-1]  # Update the ZMP time to the last value

        # Add the new positions to our trajectory
        torso_pos.append((time, target_torso_pos))
        foot_step.append((time, target_swing_pos, next_support_leg))

        # Update current positions for the next step
        current_torso = target_torso_pos
        current_swing = target_swing_pos

        # Compute the remaining steps for N future steps (1 step is the initial, then the last is to move back to center)
        for _ in range(self.n_steps - 2):
            # Calculate the next torso and swing foot positions
            target_torso_pos, target_swing_pos = self.calcTorsoFoot(
                vel_cmd, current_torso, left_is_swing)
            
            # Advance time for the next step
            time += self.t_step
            
            # Compute ZMP trajectory for this step
            temp_zmp, temp_tzmp = self.compute_zmp_trajectory(
                zmp_t, current_torso[:2], target_torso_pos[:2], current_swing[:2])

            zmp_pos.extend(temp_zmp)
            timer_count.extend(temp_tzmp)
            zmp_t = timer_count[-1]  # Update the ZMP time to the last value

            # Add the new positions to our trajectory
            torso_pos.append((time, target_torso_pos))
            foot_step.append((time, target_swing_pos, next_support_leg))

            # Toggle the support leg for the next step
            if left_is_swing:
                next_support_leg = SupportLeg.LEFT
                left_is_swing = 0
            else:
                next_support_leg = SupportLeg.RIGHT
                left_is_swing = 1

            # Update current positions for the next step
            current_torso = target_torso_pos
            current_swing = target_swing_pos

        # Add a final step to return to a balanced position
        time += self.t_step

        # Calculate the final positions for a balanced stance
        # At this point, current_swing is the swing foot from the last step calculation
        # and next_support_leg indicates which leg will be the support in the next step
        if next_support_leg == SupportLeg.LEFT:
            # Left leg will be support, right leg will be the final swing foot
            # Position the right foot at a lateral distance from the left foot
            final_swing_pos = self.transform_to_frame(
                np.array([[0], [-2 * self.y_sep], [0]], dtype=np.float32), 
                current_swing)  # Position right foot relative to left foot
            
            # Position torso halfway between the feet
            final_torso_pos = self.transform_to_frame(
                np.array([[0], [-self.y_sep], [0]], dtype=np.float32), 
                current_swing)  # Position torso between feet
        else:
            # Right leg will be support, left leg will be the final swing foot
            # Position the left foot at a lateral distance from the right foot
            final_swing_pos = self.transform_to_frame(
                np.array([[0], [2 * self.y_sep], [0]], dtype=np.float32), 
                current_swing)  # Position left foot relative to right foot
            
            # Position torso halfway between the feet
            final_torso_pos = self.transform_to_frame(
                np.array([[0], [self.y_sep], [0]], dtype=np.float32), 
                current_swing)  # Position torso between feet

        # Add the final positions to our trajectory
        torso_pos.append((time, final_torso_pos))
        foot_step.append((time, final_swing_pos, next_support_leg))

        # Compute ZMP trajectory for the final step
        temp_zmp, temp_tzmp = self.compute_zmp_trajectory(
            zmp_t, current_torso[:2], final_torso_pos[:2], current_swing[:2])
        zmp_pos.extend(temp_zmp)
        timer_count.extend(temp_tzmp)
        zmp_t = timer_count[-1]  # Update the ZMP time to the last value

        # Add a final double support phase for stability
        time += self.t_step
        next_support_leg = SupportLeg.BOTH  # Both feet on the ground
        foot_step.append((time, final_swing_pos, next_support_leg))
        torso_pos.append((time, final_torso_pos))

        # Add ZMP points at the final torso position for stability
        for _ in range(int(self.t_end // self.dt)):
            zmp_pos.append(final_torso_pos[:2])
            zmp_t += self.dt
            timer_count.append(zmp_t)

        # Store the ZMP trajectory for later use by get_zmp_reference_for_preview_control
        self.zmp_pos = zmp_pos
        self.timer_count = timer_count

        return foot_step, torso_pos, zmp_pos, timer_count

    def calculate_footsteps(self, vel_cmd: Union[List[float], Tuple[float, float, float], np.ndarray],
                          current_swing: Union[List[float], Tuple[float, float, float], np.ndarray],
                          current_support: Union[List[float], Tuple[float, float, float], np.ndarray],
                          current_torso: Union[List[float], Tuple[float, float, float], np.ndarray],
                          next_support_leg: SupportLeg) -> List[Dict]:
        """
        Calculate N consecutive steps from the given velocity commands based on current support foot.
        
        This method generates a sequence of footsteps, torso positions, and ZMP trajectory
        based on the current state and velocity commands. It handles the alternating
        support leg pattern and ensures smooth transitions between steps.
        
        Parameters
        ----------
        vel_cmd : Union[List[float], Tuple[float, float, float], np.ndarray]
            Velocity command input as [x_vel, y_vel, angular_vel]
        current_swing : Union[List[float], Tuple[float, float, float], np.ndarray]
            Current swing foot position as [x, y, theta]
        current_support : Union[List[float], Tuple[float, float, float], np.ndarray]
            Current support foot position as [x, y, theta]
        current_torso : Union[List[float], Tuple[float, float, float], np.ndarray]
            Current torso position as [x, y, theta]
        next_support_leg : SupportLeg
            Next support leg, either SupportLeg.LEFT or SupportLeg.RIGHT
            
        Returns
        -------
        step_sequence: List[Dict]
            List of dictionaries containing the following keys for each step:
            - 'time': time of the step
            - 'SF': support leg (LEFT or RIGHT)
            - 'L': current left foot position as [x, y, theta]
            - 'R': current right foot position as [x, y, theta]
            - 'C': current torso position as [x, y, theta]
            - 'next_L': next left foot position as [x, y, theta]
            - 'next_R': next right foot position as [x, y, theta]
            - 'next_C': next torso position as [x, y, theta]
            
        Raises
        ------
        ValueError
            If next_support_leg is not SupportLeg.LEFT or SupportLeg.RIGHT
        """
        # Empty method body - will be implemented later
        # For N consective steps, the outputs will be a list of {SF_i, L_i, C_i, R_i, L_{i+1}, R_{i+1}, L_{i+2}, R_{i+2}}
        # SF is support leg between [SupportLeg.LEFT, SupportLeg.RIGHT]
        # For display purpose it should log the first input data:
        #   - If next_support_leg is SupportLeg.RIGHT, then SF_0 is LEFT, else RIGHT;
        #   - if next_support_leg is RIGHT, then R_0 is the swing foot and L_0 is the support foot. Else, L_0 is the swing foot and R_0 is the support foot.
        # For i=1 to N: compute L_i, R_i, C_i, L_{i+1}, R_{i+1}, C_{i+1} using calcTorsoFoot and alternate between left and right support legs
        # SF_i is the next support leg
        # if next_support_leg is RIGHT, then L_i is the swing foot and R_i is the support foot. Else, L_i is the support foot and R_i is the swing foot
        # C_i is the current torso position
        # C_{i+1} is the next torso position
        # Validate inputs
        if next_support_leg not in [SupportLeg.LEFT, SupportLeg.RIGHT]:
            raise ValueError("next_support_leg must be SupportLeg.LEFT or SupportLeg.RIGHT for planning")
            
        # Apply velocity limits for safety
        vel_cmd = self.limit_velocity(vel_cmd)
        
        # Initialize time and containers
        time = 0.0
        step_sequence = []
        
        # Initialize left and right foot positions based on current support and swing
        left_foot = current_swing.copy() if next_support_leg == SupportLeg.RIGHT else current_support.copy()
        right_foot = current_support.copy() if next_support_leg == SupportLeg.RIGHT else current_swing.copy()
        
        # Set initial swing leg flag for calcTorsoFoot
        left_is_swing = 1 if next_support_leg == SupportLeg.RIGHT else 0
        
        # Initialize current positions
        current_SF = next_support_leg  # Current support leg is the input next_support_leg
        current_L = left_foot  # Current left foot position
        current_R = right_foot  # Current right foot position
        current_C = current_torso.copy()  # Current torso position
        
        for i in range(self.n_steps):
            # Calculate next torso and swing foot positions
            next_C, next_swing_pos = self.calcTorsoFoot(vel_cmd, current_C, left_is_swing)
            
            # Update left or right foot based on which is swinging
            if left_is_swing:
                next_L = next_swing_pos.copy()
                next_R = current_R
            else:
                next_L = current_L
                next_R = next_swing_pos.copy()
            
            # No ZMP computation in this method
            
            # Switch support leg for next iteration
            if current_SF == SupportLeg.RIGHT:
                next_SF = SupportLeg.LEFT
                left_is_swing = 0  # Right will swing next
            else:  # current_SF == SupportLeg.LEFT
                next_SF = SupportLeg.RIGHT
                left_is_swing = 1  # Left will swing next
            
            # Add this step to the sequence with both current and next positions
            step_sequence.append({
                'time': time,
                'SF': current_SF,
                'L': current_L,
                'R': current_R,
                'C': current_C,
                'next_L': next_L,
                'next_R': next_R,
                'next_C': next_C
            })
            
            # Update current positions for next iteration
            current_L = next_L
            current_R = next_R
            current_C = next_C
            current_SF = next_SF
            
            # Advance time
            time += self.t_step

        # # Add a final double support phase
        # step_sequence.append({
        #     'time': time,
        #     'SF': SupportLeg.BOTH,
        #     'L': current_L,
        #     'R': current_R,
        #     'C': current_C,
        #     'next_L': current_L,  # Same as current for final step
        #     'next_R': current_R,  # Same as current for final step
        #     'next_C': current_C   # Same as current for final step
        # })
        
        return step_sequence

    def calculate_piecewise_zmp(self, phi: float, initial_support_foot: np.ndarray, 
                               initial_torso: np.ndarray, target_torso: np.ndarray) -> np.ndarray:
        """
        Calculate ZMP position based on phase (phi) and essential foot/torso positions.
        
        This method computes a ZMP position for a given phase in the step cycle using
        the simplified phase-based approach. The method directly calculates ZMP positions
        based on the single support phase (ssp_phase) and double support phase (dsp_phase)
        configuration parameters.
        
        Parameters
        ----------
        phi : float
            Normalized time/phase within the step cycle (0 to 1)
        initial_support_foot : np.ndarray
            Initial position of the support foot as [x, y, theta] or [x, y]
        initial_torso : np.ndarray
            Initial position of the torso as [x, y, theta] or [x, y]
        target_torso : np.ndarray
            Target position of the torso as [x, y, theta] or [x, y]
            
        Returns
        -------
        np.ndarray
            ZMP position as [x, y]
        """

        # Determine ZMP position based on phase
        if phi >= 0 and phi < self.ssp_phase:
            factor = phi / self.ssp_phase
            return initial_torso[:2] * (1 - factor) + initial_support_foot[:2] * factor
        elif phi >= self.ssp_phase and phi < self.dsp_phase:
            return initial_support_foot[:2]
        elif phi >= self.dsp_phase and phi < 1:  # Changed to include phi=1.0
            factor = (1 - phi) / (1 - self.dsp_phase)
            return target_torso[:2] * (1 - factor) + initial_support_foot[:2] * factor
        elif phi >= 1:
            return target_torso[:2]
        else:
            raise ValueError("Invalid phase value")

    def generate_complete_zmp_trajectory(self, foot_step_sequence: list) -> list:
        """
        Generate a complete ZMP trajectory for an entire footstep sequence.
        
        This method processes each step in the footstep sequence and calculates the ZMP
        trajectory using the simplified calculate_piecewise_zmp method. For each step,
        it samples the ZMP positions at intervals of self.dt from t=0 to t=self.t_step.
        
        The method returns a list of trajectory segments, where each segment contains
        the time array and corresponding ZMP positions for a single step. This format
        allows for easier integration with the preview control system and visualization.
        
        Parameters
        ----------
        foot_step_sequence : list[dict]
            List of footstep data dictionaries from calculate_footsteps, each containing:
            - 'SF': SupportLeg enum indicating the support foot (LEFT, RIGHT)
            - 'L': Left foot position as [x, y, theta]
            - 'R': Right foot position as [x, y, theta]
            - 'C': Current torso position as [x, y, theta]
            - 'next_C': Next torso position as [x, y, theta]
            
        Returns
        -------
        list[list]
            List of trajectory segments, where each segment is a list containing:
            - Time array of shape (n_samples,)
            - ZMP positions array of shape (n_samples, 2) for x,y coordinates
        """
        complete_zmp_trajectory = []

        # Calculate the number of samples per step
        num_samples = int(self.t_step / self.dt) + 1

        # Process step sequences
        for step_data in foot_step_sequence:

            # Determine support foot based on support leg
            support_leg = step_data['SF']

            # Extract positions
            if support_leg == SupportLeg.LEFT:
                support_foot = step_data['L']
            elif support_leg == SupportLeg.RIGHT:
                support_foot = step_data['R']

            current_torso = step_data['C']
            next_torso = step_data['next_C']

            zmp_pos = []
            zmp_t = []
            # Generate ZMP trajectory for this step by sampling at different phases
            for j in range(num_samples):
                t = j * self.dt  # Current time within step
                phi = t / self.t_step  # Normalized time (walk phase)

                # Calculate ZMP position at this phase
                zmp_xy = self.calculate_piecewise_zmp(
                    phi,
                    support_foot,
                    current_torso,
                    next_torso
                )
                zmp_t.append(t)
                zmp_pos.append(zmp_xy)

            zmp_pos = np.array(zmp_pos).reshape(-1, 2)
            zmp_t = np.array(zmp_t)
            complete_zmp_trajectory.append([zmp_t, zmp_pos])

        return complete_zmp_trajectory
