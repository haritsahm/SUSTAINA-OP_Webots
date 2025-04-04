import os
import sys
from typing import List, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mathematics'))

import numpy as np
from mathematics.linear_algebra import get_transition_xyz

class LinkData:
    """
    Represents the data structure for a robot link, including its physical properties
    and kinematic state.
    """
    def __init__(self):
        self.name: str = ""
        
        # Link relationships
        self.parent: int = -1
        self.sibling: int = -1
        self.child: int = -1
        
        # Physical properties
        self.mass: float = 0.0
        self.relative_position: np.ndarray = get_transition_xyz(0.0, 0.0, 0.0)
        self.joint_axis: np.ndarray = get_transition_xyz(0.0, 0.0, 0.0)
        self.center_of_mass: np.ndarray = get_transition_xyz(0.0, 0.0, 0.0)
        self.inertia: np.ndarray = np.zeros((3, 3))
        
        # Joint limits
        self.joint_limit_max: float = 100.0
        self.joint_limit_min: float = -100.0
        
        # Joint state
        self.joint_angle: float = 0.0
        self.joint_velocity: float = 0.0
        self.joint_acceleration: float = 0.0
        
        # Position and orientation
        self.position: np.ndarray = get_transition_xyz(0.0, 0.0, 0.0)
        self.rotation: np.ndarray = np.eye(3)
        self.transformation: np.ndarray = np.eye(4)