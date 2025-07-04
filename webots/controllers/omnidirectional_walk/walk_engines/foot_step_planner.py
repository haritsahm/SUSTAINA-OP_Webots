import numpy as np
from typing import Tuple, List, Union, Dict
from omegaconf import DictConfig
import logging
from enum import Enum

logger = logging.getLogger(__name__)

CONST_G = 9.806

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
        self.dt = planner_config.get('dt', 0.01)  # Sampling time (seconds)
        
        # Phase parameters
        self.ssp_phase = planner_config.get('ssp_phase', 0.2)  # Single support phase ratio
        self.dsp_phase = 1.0 - self.ssp_phase  # Double support phase ratio

        self.ssp_time = self.ssp_phase * self.t_step
        self.dsp_time = self.dsp_phase * self.t_step
            
        self.y_sep = planner_config.get('foot_separation', 0.054)  # Half hip width (meters)
        self.com_height = planner_config.get('com_height', 0.18)  # CoM height (meters)
        self.t_zmp = np.sqrt(self.com_height / CONST_G)
        self.phi_zmp = self.t_zmp / self.t_step

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

    def calcHfunc(self, t_time, t_norm):
        """Horizontal Trajectory function

        Reference: Maximo, Marcos - Omnidirectional ZMP-Based Walking for Humanoid
        Section 3.3 - CoM Trajectory Using 3D-LIPM

        Calculate the trajectory using C1-continuous spline interpolation eq. (3.36)

        Parameters
        ----------
        t_time : float
            The time since the beginning of the step
        t_norm : float
            Normalized time since the beginning of the step [0, 1]

        Returns
        -------
        float
            Value of interpolation at time t_time
        """
        h_func = 0
        if t_norm < self.ssp_phase:
            h_func = 0
        elif t_norm >= self.ssp_phase and t_norm < self.dsp_phase:
            h_func = 0.5 * \
                (1 - np.cos(np.pi * ((t_time - self.ssp_time) / (self.dsp_time - self.ssp_time))))
        elif t_norm >= self.dsp_phase:
            h_func = 1
        return h_func

    def calcVfunc(self, t_time, t_norm):
        """Vertical Trajectory function

        Reference: Maximo, Marcos - Omnidirectional ZMP-Based Walking for Humanoid
        Section 3.3 - CoM Trajectory Using 3D-LIPM

        Calculate the trajectory using C1-continuous spline interpolation eq. (3.39)

        Parameters
        ----------
        t_time : float
            The time since the beginning of the step
        t_norm : float
            Normalized time since the beginning of the step [0, 1]

        Returns
        -------
        float
            Value of interpolation at time t
        """

        v_func = 0
        if t_norm < self.ssp_phase or t_norm >= self.dsp_phase:
            v_func = 0
        elif t_norm >= self.ssp_phase and t_norm < self.dsp_phase:
            v_func = 0.5 * (1 - np.cos(2 * np.pi * ((t_norm -
                            self.dsp_phase) / (self.dsp_phase - self.ssp_phase))))
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
            diff_x = (T_k2_rel_P - T_k1_rel_P)[0]
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
        S_k1_rel_P[0] = T_k1_rel_P[0] + (float(diff_x) / 2)
        
        # Step 6: Transform the swing foot pose from P frame to support foot frame
        S_k1_rel_S_k = self.pose_global2d(S_k1_rel_P, P)
        S_k1_rel_S_k[2] = P[2]  # Set the orientation to match the reference frame

        return T_k1, S_k1_rel_S_k

    def calculate_single_step(self, vel_cmd: Union[List[float], Tuple[float, float, float], np.ndarray],
                          left_foot: Union[List[float], Tuple[float, float, float], np.ndarray],
                          right_foot: Union[List[float], Tuple[float, float, float], np.ndarray],
                          current_torso: Union[List[float], Tuple[float, float, float], np.ndarray],
                          next_support_leg: SupportLeg) -> Dict:
        """Calculate a single step based on current global positions and next support leg.

        This method computes the next step positions given the current global positions of
        the feet and torso, along with the desired velocity command. All input and output
        positions are in the global frame.

        Parameters
        ----------
        vel_cmd : array-like, shape (3,)
            Velocity command [vx, vy, omega] in m/s and rad/s
        left_foot : array-like, shape (3,)
            Current left foot pose [x, y, theta] in global frame
        right_foot : array-like, shape (3,)
            Current right foot pose [x, y, theta] in global frame
        current_torso : array-like, shape (3,)
            Current torso pose [x, y, theta] in global frame
        next_support_leg : SupportLeg
            Which leg will be the support leg (LEFT or RIGHT)

        Returns
        -------
        dict
            Dictionary containing:
            - SF: Current support leg (SupportLeg enum)
            - L, R, C: Current left, right, torso poses in global frame
            - next_L, next_R, next_C: Next left, right, torso poses in global frame
            - next_SF: Next support leg (SupportLeg enum)

        Notes
        -----
        The method follows these steps:
        1. Determine support foot and transform positions to support foot frame
        2. Generate next torso and swing foot positions using calcTorsoFoot
        3. Transform all positions back to global frame
        4. Return current and next positions in global frame
        """
        if next_support_leg not in [SupportLeg.LEFT, SupportLeg.RIGHT]:
            raise ValueError("next_support_leg must be SupportLeg.LEFT or SupportLeg.RIGHT for planning")
            
        vel_cmd = self.limit_velocity(vel_cmd)
        
        if next_support_leg == SupportLeg.RIGHT:
            current_support = right_foot.copy()
            current_swing_local = self.pose_relative2d(left_foot, current_support)
        else:
            current_support = left_foot.copy()
            current_swing_local = self.pose_relative2d(right_foot, current_support)
        current_torso_local = self.pose_relative2d(current_torso, current_support)

        left_is_swing = 1 if next_support_leg == SupportLeg.RIGHT else 0
        next_C_local, next_swing_local = self.calcTorsoFoot(vel_cmd, current_torso_local, left_is_swing)
        next_torso = self.transform_to_frame(next_C_local, current_support)
        next_swing = self.transform_to_frame(next_swing_local, current_support)

        step_data = {
            'SF': next_support_leg,  # current support foot
            'L': left_foot.ravel(),
            'R': right_foot.ravel(),
            'C': current_torso.ravel(),
            'next_L': next_swing.ravel() if next_support_leg == SupportLeg.RIGHT else current_support.ravel(),
            'next_R': current_support.ravel() if next_support_leg == SupportLeg.RIGHT else next_swing.ravel(),
            'next_C': next_torso.ravel(),
            'next_SF': SupportLeg.LEFT if next_support_leg == SupportLeg.RIGHT else SupportLeg.RIGHT
        }

        return step_data

    def calculate_piecewise_zmp(self, phi: float, initial_support_foot: np.ndarray, 
                               initial_torso: np.ndarray, target_torso: np.ndarray) -> np.ndarray:
        """
        Calculate ZMP position based on phase (phi) and essential foot/torso positions.
        
        This method computes a ZMP position for a given phase in the step cycle using
        the simplified phase-based approach. The ZMP trajectory is defined piecewise:
        
        For 0 ≤ φ < ssp_phase (Initial single support to double support transition):
            p(φ) = initial_torso·(1 - factor) + initial_support_foot·factor
            where factor = φ/ssp_phase
        
        For ssp_phase ≤ φ < dsp_phase (Double support phase):
            p(φ) = initial_support_foot
        
        For dsp_phase ≤ φ < 1 (Double support to final single support transition):
            p(φ) = target_torso·(1 - factor) + initial_support_foot·factor
            where factor = (1 - φ)/(1 - dsp_phase)
        
        For φ ≥ 1 (Beyond step cycle):
            p(φ) = target_torso
        
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
        initial_support_foot_xy = initial_support_foot[:2].reshape(2, 1)
        initial_torso_xy = initial_torso[:2].reshape(2, 1)
        target_torso_xy = target_torso[:2].reshape(2, 1)

        # Determine ZMP position based on phase
        if phi >= 0 and phi < self.ssp_phase:
            factor = phi / self.ssp_phase
            return initial_torso_xy * (1 - factor) + initial_support_foot_xy * factor
        elif phi >= self.ssp_phase and phi < self.dsp_phase:
            return initial_support_foot_xy
        elif phi >= self.dsp_phase and phi <= 1:  # Changed to include phi=1.0
            factor = (1 - phi) / (1 - self.dsp_phase)
            return target_torso_xy * (1 - factor) + initial_support_foot_xy * factor
        else:
            raise ValueError("Invalid phase value")

    def compute_boundary_com_constraints(self, initial_support_foot: np.ndarray, 
                         initial_torso: np.ndarray, 
                         target_torso: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute coefficients for the homogeneous solution of CoM boundary constraints.

        This method calculates the coefficients aP and aN for the homogeneous solution
        of the Linear Inverted Pendulum Model (LIPM) boundary value problem. These
        coefficients ensure smooth CoM trajectory transitions between support phases.

        Parameters
        ----------
        initial_support_foot : array-like, shape (3,) or (2,)
            Initial support foot pose [x, y, theta] or position [x, y] in global frame
        initial_torso : array-like, shape (3,) or (2,)
            Initial torso pose [x, y, theta] or position [x, y] in global frame
        target_torso : array-like, shape (3,) or (2,)
            Target torso pose [x, y, theta] or position [x, y] in global frame

        Returns
        -------
        aP : ndarray, shape (2,)
            Coefficient for the positive exponential term (exp(t/T))
        aN : ndarray, shape (2,)
            Coefficient for the negative exponential term (exp(-t/T))
        M_i : ndarray, shape (2,)
            Initial transition vector for the step
        N_i : ndarray, shape (2,)
            Final transition vector for the step

        Notes
        -----
        The method uses the following steps:
        1. Extract 2D positions from input poses
        2. Compute transition vectors M_i and N_i for boundary conditions
        3. Calculate transition terms using time constants and phase durations
        4. Solve for coefficients aP and aN using exponential terms

        The coefficients satisfy the boundary conditions:
        - At phi=0: Initial CoM position and velocity
        - At phi=1: Final CoM position and velocity
        """
        support_foot_xy = np.array(initial_support_foot[:2]).reshape(2, 1)
        initial_com_xy = np.array(initial_torso[:2]).reshape(2, 1)
        target_com_xy = np.array(target_torso[:2]).reshape(2, 1)
        
        alpha = 1.0 / self.phi_zmp
        
        M_i = (support_foot_xy - initial_com_xy) / self.ssp_phase
        N_i = -(support_foot_xy - target_com_xy) / (1 - self.dsp_phase)
                
        ssp_ratio = self.ssp_phase / self.phi_zmp
        term_0_factor = -ssp_ratio + np.sinh(ssp_ratio)
        trans_term_0 = -M_i * self.t_zmp * term_0_factor
        
        phase_diff = (1 - self.dsp_phase) / self.phi_zmp
        term_1_factor = phase_diff - np.sinh(phase_diff)
        trans_term_1 = -N_i * self.t_zmp * term_1_factor
        
        exp_alpha = np.exp(alpha)
        exp_neg_alpha = np.exp(-alpha)
        denom = exp_alpha - exp_neg_alpha
        
        aP = (trans_term_1 - trans_term_0 * exp_neg_alpha ) / denom
        aN = (trans_term_0 * exp_alpha - trans_term_1) / denom
        
        return aP, aN, M_i, N_i

    def calculate_analytical_com(self, phi: float, zmp_pos: np.ndarray, 
                                M_transition: np.ndarray, N_transition: np.ndarray,
                                aP: np.ndarray, aN: np.ndarray) -> np.ndarray:
        """Calculate analytical CoM position using LIPM solution.

        Computes the Center of Mass (CoM) position at a given phase using the Linear
        Inverted Pendulum Model (LIPM). The solution combines:
        1. Particular solution: Current ZMP position
        2. Homogeneous solution: exp(±t/T) terms with coefficients aP, aN
        3. Transition terms: Additional dynamics during support phase changes

        Parameters
        ----------
        phi : float
            Current phase in the step cycle (0 ≤ φ ≤ 1)
        zmp_pos : ndarray, shape (2,)
            Current Zero Moment Point (ZMP) position [x, y] in global frame
        M_transition : ndarray, shape (2,)
            Initial transition vector from CoM to support foot
        N_transition : ndarray, shape (2,)
            Final transition vector from support foot to target CoM
        aP : ndarray, shape (2,)
            Coefficient for positive exponential term exp(t/T)
        aN : ndarray, shape (2,)
            Coefficient for negative exponential term exp(-t/T)

        Returns
        -------
        ndarray, shape (2,)
            Computed CoM position [x, y] in global frame

        Notes
        -----
        The solution structure varies by phase:
        - Initial SSP (0 ≤ φ < ssp_phase):
          CoM = ZMP + aP·exp(t/T) + aN·exp(-t/T) + M_transition·transition_term
        - DSP (ssp_phase ≤ φ < dsp_phase):
          CoM = ZMP + aP·exp(t/T) + aN·exp(-t/T)
        - Final SSP (dsp_phase ≤ φ < 1):
          CoM = ZMP + aP·exp(t/T) + aN·exp(-t/T) + N_transition·transition_term
        - Beyond step (φ ≥ 1): CoM = ZMP
        """
        exp_p = np.exp(phi / self.phi_zmp)
        exp_n = np.exp(-phi / self.phi_zmp)
        homogeneous_term = aP * exp_p + aN * exp_n

        if phi >= 0 and phi < self.ssp_phase:
            phase_diff = phi - self.ssp_phase
            transition_term = M_transition * self.t_zmp * (
                (phase_diff / self.phi_zmp) - np.sinh(phase_diff / self.phi_zmp)
            )
            return zmp_pos + homogeneous_term + transition_term
        elif phi >= self.ssp_phase and phi < self.dsp_phase:
            return zmp_pos + homogeneous_term
        elif phi >= self.dsp_phase and phi < 1:
            phase_diff = phi - self.dsp_phase
            transition_term = N_transition * self.t_zmp * (
                (phase_diff / self.phi_zmp) - np.sinh(phase_diff / self.phi_zmp)
            )
            return zmp_pos + homogeneous_term + transition_term

    def generate_complete_zmp_trajectory(self, step_data: Dict) -> List[Dict[str, Union[float, List[float]]]]:
        """Generate complete Zero Moment Point trajectory for a single step.

        Computes the ZMP trajectory by sampling positions at regular intervals
        throughout the step cycle. The trajectory is computed in the global frame
        using piecewise ZMP calculations based on the walking phase.

        Parameters
        ----------
        step_data : dict
            Step data containing:
            - SF: Support foot leg (LEFT/RIGHT)
            - L: Left foot pose [x, y, theta]
            - R: Right foot pose [x, y, theta]
            - C: Current torso pose [x, y, theta]
            - next_C: Next torso pose [x, y, theta]
            All poses are in global frame.

        Returns
        -------
        list of dict
            List of trajectory points, each containing:
            - time: Time from step start [s]
            - phase: Normalized phase [0,1]
            - position: ZMP position [x, y] in global frame

        Notes
        -----
        The ZMP trajectory is sampled at dt intervals over the step duration.
        Positions are computed using calculate_piecewise_zmp which handles
        the different walking phases:
        - Initial single support (0 ≤ φ < ssp_phase)
        - Double support (ssp_phase ≤ φ < dsp_phase)
        - Final single support (dsp_phase ≤ φ < 1)
        """
        zmp_trajectory = []
    
        # Extract support foot based on support leg
        support_leg = step_data['SF']
        if support_leg == SupportLeg.LEFT:
            support_foot_xy = step_data['L'][:2].reshape(2, 1)
        elif support_leg == SupportLeg.RIGHT:
            support_foot_xy = step_data['R'][:2].reshape(2, 1)
        else:
            raise ValueError(f"Unknown support leg type: {support_leg}")
        
        initial_torso_xy = step_data['C'][:2].reshape(2, 1)
        target_torso_xy = step_data['next_C'][:2].reshape(2, 1)

        num_samples = int(self.t_step / self.dt) + 1
        for j in range(num_samples):
            t = j * self.dt
            phi = t / self.t_step
            
            zmp_global = self.calculate_piecewise_zmp(
                phi=phi,
                initial_support_foot=support_foot_xy,
                initial_torso=initial_torso_xy,
                target_torso=target_torso_xy
            )
            
            zmp_trajectory.append({
                'time': float(t),
                'phase': float(phi),
                'position': zmp_global.ravel().tolist()
            })
        
        return zmp_trajectory

    def generate_complete_com_trajectory(self, step_data: Dict, with_zmp: bool = False) -> List[Dict[str, Union[float, List[float]]]]:
        """Generate complete Center of Mass trajectory for a single step.

        Computes the CoM trajectory using the Linear Inverted Pendulum Model (LIPM)
        analytical solution. For each time sample, calculates both the ZMP and CoM
        positions in the global frame.

        Parameters
        ----------
        step_data : dict
            Step data containing:
            - SF: Support foot leg (LEFT/RIGHT)
            - L: Left foot pose [x, y, theta]
            - R: Right foot pose [x, y, theta]
            - C: Current torso pose [x, y, theta]
            - next_C: Next torso pose [x, y, theta]
            All poses are in global frame
        with_zmp : bool, optional
            If True, includes ZMP positions in trajectory points, by default False

        Returns
        -------
        list of dict
            List of trajectory points, each containing:
            - time: Time from step start [s]
            - phase: Normalized phase [0,1]
            - position: CoM position [x, y] in global frame
            - zmp_pos: ZMP position [x, y] if with_zmp=True

        Notes
        -----
        The method uses these steps:
        1. Compute boundary constraints (aP, aN) and transition vectors
        2. Sample trajectory points at dt intervals
        3. For each point:
           - Calculate ZMP position using piecewise function
           - Calculate CoM position using analytical solution
        """
        com_trajectory = []
        
        support_leg = step_data['SF']
        if support_leg == SupportLeg.LEFT:
            support_foot_xy = step_data['L'][:2].reshape(2, 1)
        elif support_leg == SupportLeg.RIGHT:
            support_foot_xy = step_data['R'][:2].reshape(2, 1)
        else:
            raise ValueError(f"Unknown support leg type: {support_leg}")
        
        initial_torso_xy = step_data['C'][:2].reshape(2, 1)
        target_torso_xy = step_data['next_C'][:2].reshape(2, 1)

        # Get boundary constraints and transition vectors
        aP, aN, M_i, N_i = self.compute_boundary_com_constraints(
            initial_support_foot=support_foot_xy,
            initial_torso=initial_torso_xy,
            target_torso=target_torso_xy
        )

        num_samples = int(self.t_step / self.dt) + 1
        for j in range(num_samples):
            t = j * self.dt
            phi = t / self.t_step
            
            # Get current ZMP position
            zmp_global = self.calculate_piecewise_zmp(
                phi=phi,
                initial_support_foot=support_foot_xy,
                initial_torso=initial_torso_xy,
                target_torso=target_torso_xy
            )
            
            # Calculate CoM position
            if phi >= 0 and phi < 1.0:
                com_global = self.calculate_analytical_com(
                    phi=phi,
                    zmp_pos=zmp_global,
                    M_transition=M_i,
                    N_transition=N_i,
                    aP=aP,
                    aN=aN
                )
            else:
                com_global = np.asarray(target_torso_xy).ravel()
            
            # Create trajectory point
            point = {
                'time': float(t),
                'phase': float(phi),
                'position': com_global.ravel().tolist()
            }
            
            # Add ZMP position if requested
            if with_zmp:
                point['zmp_position'] = zmp_global.ravel().tolist()
            
            com_trajectory.append(point)        
        return com_trajectory
