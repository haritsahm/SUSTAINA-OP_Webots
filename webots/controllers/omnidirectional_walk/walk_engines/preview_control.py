"""
  Source: yiqin/Preview-Control-Motion-Planning-in-Humanoid
  
  Module to implement a ZMP preview controller.
  This module is based on the papers:
  [1] Kajita, Shuuji, et al. "Biped walking pattern generation by
      using preview control of zero-moment point." Proc. IEEE Int'l
      Conf. on Robotics and Automation (ICRA), IEEE 2003.
  [2] Park, Jonghoon, and Youngil Youm. "General ZMP preview control
      for bipedal walking." Proc. IEEE Int'l Conf. on Robotics and
      Automation (ICRA), IEEE 2007
"""
import numpy as np
import scipy
from typing import Tuple
from omegaconf import DictConfig


class PreviewControl():
    """ZMP Preview Controller for humanoid walking pattern generation in the 2D plane.
    
    This class implements the preview control method for generating stable walking patterns
    based on a reference ZMP trajectory. It uses a discrete-time state-space model of the
    linear inverted pendulum to predict and control the Center of Mass (CoM) motion.
    
    The controller operates in the 2D (x,y) plane only, with a static z-height set by the com_height
    parameter. Separate controllers should be used for the x and y dimensions, as they are treated
    independently.
    
    The controller minimizes a cost function that penalizes ZMP tracking error and control effort,
    while looking ahead at future reference ZMP positions to generate optimal control inputs.

    Parameters
    ----------
    config : DictConfig
        Configuration parameters from config.yaml
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the Preview Controller with the given configuration.
        
        This sets up the state-space model and computes the optimal gains
        for the preview controller based on the specified parameters in config.yaml.
        
        Parameters
        ----------
        config : DictConfig
            Configuration parameters from config.yaml
        """
        # Extract configuration parameters
        if 'preview_control' not in config:
            raise ValueError("Missing 'preview_control' section in configuration")
            
        preview_config = config.preview_control
        
        # Store configuration parameters
        self.dt = preview_config.get('dt', 0.01)  # Sampling time (seconds)
        preview_t = preview_config.get('preview_t', 2.0)  # Preview horizon time (seconds)
        self.n_preview = int(preview_t // self.dt)  # Number of preview steps
        
        # Store cost function weights
        self._R = preview_config.get('R', 1e-6)
        self._Qe = preview_config.get('Qe', 1.0)
        self._Qdpos = preview_config.get('Qdpos', 0.0)
        self._Qdvel = preview_config.get('Qdvel', 0.0)
        self._Qdaccel = preview_config.get('Qdaccel', 0.0)
        
        # Gravitational acceleration (m/s²)
        self.gravity = 9.8
        
        # Initialize private com_height variable
        self._com_height = None
        
        # Set com_height which will trigger the calculation of the state-space model and gains
        self.com_height = preview_config.get('com_height', 0.23)
    
    @property
    def com_height(self) -> float:
        """Get the current CoM height."""
        return self._com_height
    
    @com_height.setter
    def com_height(self, value: float):
        """Set the CoM height and recalculate the state-space model and gains.
        
        Parameters
        ----------
        value : float
            New CoM height in meters
        """
        if value <= 0:
            raise ValueError("CoM height must be positive")
            
        # Only recalculate if the value has changed
        if self._com_height != value:
            self._com_height = value
            self._calculate_state_space_model()
    
    def _calculate_state_space_model(self):
        """Calculate the state-space model and gains based on current parameters.
        
        This method is called automatically when com_height is changed.
        """
        # State-space model for the CoM dynamics (Linear Inverted Pendulum Model)
        # Note: This model operates in a single dimension (x or y) of the 2D plane
        # The controller should be instantiated twice - once for x and once for y
        # State vector: [position, velocity, acceleration]
        # x_{k+1} = A*x_k + B*u_k
        # State transition matrix A
        self.A = np.matrix([
            [1.0, self.dt, self.dt**2 / 2],  # Position update
            [0.0, 1.0, self.dt],             # Velocity update
            [0.0, 0.0, 1.0]                  # Acceleration update
        ])
        
        # Control input matrix B
        self.B = np.matrix([self.dt**3 / 6, self.dt**2 / 2, self.dt]).reshape((3, 1))

        # Output matrix C for ZMP (relates state to ZMP position)
        # ZMP = position - height/gravity * acceleration
        self.C = np.matrix([1, 0, -self._com_height / self.gravity]).reshape((1, 3))

        # Augmented state-space model including the error state
        # Augmented state: [error, position, velocity, acceleration]
        augmented_A = np.vstack((
            np.hstack((np.eye(1), self.C * self.A)),  # Error dynamics
            np.hstack((np.zeros((3, 1)), self.A))     # State dynamics
        ))

        augmented_B = np.vstack((self.C * self.B, self.B))
        
        # Cost function weights
        control_weight = self._R  # Weight on control input
        state_weights = np.diag([self._Qe, self._Qdpos, self._Qdvel, self._Qdaccel])  # Weights on state variables
        
        # Solve the discrete-time algebraic Riccati equation for optimal control
        riccati_solution = scipy.linalg.solve_discrete_are(
            augmented_A, augmented_B, state_weights, control_weight)

        # Calculate the optimal feedback gain
        # (R + B^T * P * B)^-1 * B^T * P * A
        denominator = (control_weight + augmented_B.T * riccati_solution * augmented_B)[0, 0]
        if abs(denominator) < 1e-10:
            # Add numerical stability check
            raise ValueError("Numerical instability in controller calculation")
            
        gain_factor = 1.0 / denominator
        feedback_gain = gain_factor * augmented_B.T * riccati_solution * augmented_A
        
        # Extract gains for error and state feedback
        error_gain = feedback_gain[0, 0]  # Gain for accumulated error
        state_gain = feedback_gain[0, 1:4]  # Gains for position, velocity, acceleration

        # Calculate preview gains (gains for future reference ZMP positions)
        # Based on Theorem 1 from the paper
        closed_loop_A = augmented_A - augmented_B * feedback_gain
        preview_term = -closed_loop_A.T * riccati_solution * np.matrix([[1, 0, 0, 0]]).T * state_weights[0, 0]

        # Initialize preview gains array
        preview_gains = np.zeros(self.n_preview)
        preview_gains[0] = -error_gain  # First preview gain

        # Calculate remaining preview gains
        for i in range(1, self.n_preview):
            preview_gains[i] = (gain_factor * augmented_B.T * preview_term)[0, 0]
            preview_term = closed_loop_A.T * preview_term

        # Store the computed gains for use in control calculations
        self.G = preview_gains  # Preview gains for future ZMP references
        self.Ks = error_gain    # Feedback gain for accumulated error
        self.Kx = state_gain    # Feedback gains for state variables

    def init_state_err(self, pos: float = 0, vel: float = 0, accel: float = 0, e: float = 0) -> Tuple[np.matrix, float]:
        """Initialize a state_err object for a single dimension (x or y) of the 2D plane.
        
        This method creates a state object that holds position, velocity, acceleration 
        as a vector, and accumulated ZMP tracking error as a scalar. For a complete 2D 
        controller, you should initialize separate state objects for both x and y dimensions.
        
        Parameters
        ----------
        pos : float, optional
            Initial position in a single dimension (m), by default 0
        vel : float, optional
            Initial velocity in a single dimension (m/s), by default 0
        accel : float, optional
            Initial acceleration in a single dimension (m/s²), by default 0
        e : float, optional
            Initial accumulated ZMP tracking error for this dimension, by default 0
            
        Returns
        -------
        Tuple[np.matrix, float]
            State vector (position, velocity, acceleration) and accumulated error for a single dimension
        """
        X = np.matrix([[pos, vel, accel]]).T

        return (X, e)

    def update_state_err(self, state_err: Tuple[np.matrix, float], zmp_ref: np.ndarray) -> Tuple[Tuple[np.matrix, float], float, float]:
        """Run the controller to compute the next state based on the current state and reference ZMP trajectory.
        
        This method operates on a single dimension (x or y) of the 2D plane. To control both dimensions,
        this method should be called separately for each dimension with the appropriate state and reference.
        
        The state_err argument should be a (state, error) tuple (as returned by init_state_err() or this
        function). The zmp_ref argument should be an array of future desired ZMP positions. If zmp_ref 
        is of less length than the lookahead window size, the reference trajectory is padded with
        repeats of the last element.

        This function returns three values: the new state_err after the control is executed, 
        the new ZMP position, and the generated control input.

        Reference: Design of an optimal controller for a discrete-time system subject to previewable demand
        
        Parameters
        ----------
        state_err : Tuple[np.matrix, float]
            Current state and accumulated error as returned by init_state_err() or this function
            for a single dimension (x or y)
        zmp_ref : np.ndarray
            Array of future desired ZMP positions for a single dimension
            
        Returns
        -------
        Tuple[Tuple[np.matrix, float], float, float]
            New state and error, new ZMP position, and control input for the specified dimension
        """

        # Extract and prepare future ZMP reference trajectory
        zmp_ref_array = np.array(zmp_ref).flatten()
        n_ref_points = len(zmp_ref_array)
        
        # If reference trajectory is shorter than preview window, pad with last value
        if n_ref_points < self.n_preview:
            n_padding = self.n_preview - n_ref_points
            preview_window = np.hstack((zmp_ref_array[0:], np.ones(n_padding) * zmp_ref_array[-1]))
        else:
            preview_window = zmp_ref_array[0:self.n_preview]

        # Extract current state and accumulated error
        current_state, accumulated_error = state_err

        # Calculate control input using preview control law
        # u = -Ks*e - Kx*x - Σ(G_i * p_{k+i})
        control_input = -self.Ks * accumulated_error - self.Kx * current_state - np.dot(self.G, preview_window)

        # Update state using state-space model and compute resulting ZMP
        new_state = self.A * current_state + self.B * control_input
        current_zmp = self.C * new_state
        
        # Update accumulated error (discrete integral of tracking error)
        new_accumulated_error = accumulated_error + current_zmp - zmp_ref_array[0]

        # Return new state and error, current ZMP position, and control input
        return (new_state, new_accumulated_error), current_zmp, control_input
