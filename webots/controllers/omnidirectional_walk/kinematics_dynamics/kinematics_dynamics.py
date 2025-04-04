import sys
import os
import logging
import numpy as np
from typing import List, Dict
from .link_data import LinkData

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mathematics'))
from mathematics.linear_algebra import (
    get_inertia_xyz,
    get_transition_xyz,
    calc_rodrigues,
    calc_hatto,
    calc_cross,
    convert_rot_to_omega,
)

MAX_JOINT_ID = 19
ALL_JOINT_ID = 36

MAX_ARM_ID = 3
MAX_LEG_ID = 6

ID_HEAD_END = 19
ID_COB = 0
ID_TORSO = 0

ID_R_ARM_START = 1
ID_L_ARM_START = 2
ID_R_ARM_END = 21
ID_L_ARM_END = 22

ID_R_LEG_START = 7
ID_L_LEG_START = 8
ID_R_LEG_END = 31
ID_L_LEG_END = 30

logger = logging.getLogger(__name__)

class KinematicsDynamics:
    """Main class for robot kinematics and dynamics calculations"""
    
    def __init__(self):
        self.link_data: Dict[int, LinkData] = {}
        
    @staticmethod
    def get_sign(num: float) -> float:
        """
        Returns the sign of a number
        Args:
            num: Input number
        Returns:
            -1.0 if num < 0, 1.0 otherwise
        """
        return -1.0 if num < 0 else 1.0
        
    def initialize(self, tree: str = "WholeBody"):
        """
        Initialize the kinematic structure
        Args:
            tree: Kinematic tree type (default: "WholeBody")
        """
        # Initialize link data
        for link_id in range(ALL_JOINT_ID + 1):
            self.link_data[link_id] = LinkData()

        if tree == "WholeBody":
            self._initialize_whole_body()

    def _initialize_whole_body(self):
        """Initialize whole body kinematic structure"""
        # Base link
        self.link_data[0].name = "base"
        self.link_data[0].parent = -1
        self.link_data[0].sibling = -1
        self.link_data[0].child = 19
        self.link_data[0].mass = 1.758639
        self.link_data[0].relative_position = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[0].joint_axis = np.zeros(3)
        self.link_data[0].center_of_mass = get_transition_xyz(-0.02475, 0.000537, 0.068556)
        self.link_data[0].joint_limit_max = 100.0
        self.link_data[0].joint_limit_min = -100.0
        self.link_data[0].inertia = get_inertia_xyz(0.009982335, 0.0, 0.0, 0.007902445, 0.0, 0.00631869)

        self.link_data[34].name = "imu"
        self.link_data[34].parent = 0
        self.link_data[34].sibling = -1
        self.link_data[34].child = -1
        self.link_data[34].mass = 0.0
        self.link_data[34].relative_position = get_transition_xyz(-0.005, 0.01682, 0.003)
        self.link_data[34].joint_axis = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[34].center_of_mass = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[34].joint_limit_max = 100.0
        self.link_data[34].joint_limit_min = -100.0
        self.link_data[34].inertia = get_inertia_xyz(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Initialize the rest of the body parts (head, arms, legs)
        self._initialize_head()
        self._initialize_right_arm()
        self._initialize_left_arm()
        self._initialize_right_leg()
        self._initialize_left_leg()

    def _initialize_head(self):
        """Initialize head links"""
        # head_pan
        self.link_data[19].name = "head_yaw_joint"
        self.link_data[19].parent = 0
        self.link_data[19].sibling = 1
        self.link_data[19].child = 23
        self.link_data[19].mass = 0.011759436
        self.link_data[19].relative_position = get_transition_xyz(-0.001, 0.0, 0.1365)
        self.link_data[19].joint_axis = get_transition_xyz(0.0, 0.0, 1.0)
        self.link_data[19].center_of_mass = get_transition_xyz(0.002327479, 0, 0.008227847)
        self.link_data[19].joint_limit_max = 0.75 * np.pi
        self.link_data[19].joint_limit_min = -0.75 * np.pi
        self.link_data[19].inertia = get_inertia_xyz(0.000126538, 0.00000, 0.00000, 0.000131085, 0.00000, 0.000071725)
        
        # # head_tilt (No head tilt in sustaina-op)
        # self.link_data[20].name = "head_tilt"
        # self.link_data[20].parent = 19
        # self.link_data[20].sibling = -1
        # self.link_data[20].child = -1
        # self.link_data[20].mass = 0.13630649
        # self.link_data[20].relative_position = get_transition_xyz(0.01, 0.019, 0.0285)
        # self.link_data[20].joint_axis = get_transition_xyz(0.0, -1.0, 0.0)
        # self.link_data[20].center_of_mass = get_transition_xyz(0.002298411, -0.018634079, 0.027696734)
        # self.link_data[20].joint_limit_max = 0.5 * np.pi
        # self.link_data[20].joint_limit_min = -0.5 * np.pi
        # self.link_data[20].inertia = get_inertia_xyz(0.00113, 0.00001, -0.00005, 0.00114, 0.00002, 0.00084)

        # camera
        self.link_data[23].name = "camera"
        self.link_data[23].parent = 19
        self.link_data[23].sibling = -1
        self.link_data[23].child = -1
        self.link_data[23].mass = 0.0
        self.link_data[23].relative_position = get_transition_xyz(0.036495, 0, 0.063504)
        self.link_data[23].joint_axis = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[23].center_of_mass = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[23].joint_limit_max = 100.0
        self.link_data[23].joint_limit_min = -100.0
        self.link_data[23].inertia = get_inertia_xyz(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _initialize_right_arm(self):
        """Initialize right arm links"""
        # right arm shoulder pitch
        self.link_data[1].name = "right_shoulder_pitch_joint"
        self.link_data[1].parent = 0
        self.link_data[1].sibling = 2
        self.link_data[1].child = 3
        self.link_data[1].mass = 0.029974
        self.link_data[1].relative_position = get_transition_xyz(-0.005, -0.0785, 0.148)
        self.link_data[1].joint_axis = get_transition_xyz(0.0, 1.0, 0.0)
        self.link_data[1].center_of_mass = get_transition_xyz(0, -0.01004, -0.005235)
        self.link_data[1].joint_limit_max = 0.75 * np.pi
        self.link_data[1].joint_limit_min = -0.75 * np.pi
        self.link_data[1].inertia = get_inertia_xyz(0.000009025, 0.0, 0.0, 0.000016881, 0.0, 0.000011316)
        
        # right arm shoulder roll
        self.link_data[3].name = "right_shoulder_roll_joint"
        self.link_data[3].parent = 1
        self.link_data[3].sibling = -1
        self.link_data[3].child = 5
        self.link_data[3].mass = 0.433169
        self.link_data[3].relative_position = get_transition_xyz(0, -0.0308, -0.025)
        self.link_data[3].joint_axis = get_transition_xyz(1.0, 0.0, 0.0)
        self.link_data[3].center_of_mass = get_transition_xyz(0.000415, -0.005221, -0.047591)
        self.link_data[3].joint_limit_max = 0.0
        self.link_data[3].joint_limit_min = -np.pi
        self.link_data[3].inertia = get_inertia_xyz(0.001462386, 0.0, 0.0, 0.001573653, 0.0, 0.000250070)
        
        # right arm elbow
        self.link_data[5].name = "right_elbow_pitch_joint"
        self.link_data[5].parent = 3
        self.link_data[5].sibling = -1
        self.link_data[5].child = 21
        self.link_data[5].mass = 0.137614
        self.link_data[5].relative_position = get_transition_xyz(0, 0, -0.145)
        self.link_data[5].joint_axis = get_transition_xyz(0.0, 1.0, 0.0)
        self.link_data[5].center_of_mass = get_transition_xyz(0, -0.011654, -0.083221)
        self.link_data[5].joint_limit_max = 0.75 * np.pi
        self.link_data[5].joint_limit_min = -0.75 * np.pi
        self.link_data[5].inertia = get_inertia_xyz(0.000397774, 0.0, 0.0, 0.000391749, 0.0, 0.000056957)
        
        self.link_data[21].name = "right_arm_end_joint"
        self.link_data[21].parent = 5
        self.link_data[21].sibling = -1
        self.link_data[21].child = -1
        self.link_data[21].mass = 0.137614
        self.link_data[21].relative_position = get_transition_xyz(0.0, -0.02375, -0.182)
        self.link_data[21].joint_axis = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[21].center_of_mass = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[21].joint_limit_max = 100.0
        self.link_data[21].joint_limit_min = -100.0
        self.link_data[21].inertia = get_inertia_xyz(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
    def _initialize_left_arm(self):
        """Initialize left arm links"""
        # left arm shoulder pitch
        self.link_data[2].name = "left_shoulder_pitch_joint"
        self.link_data[2].parent = 0
        self.link_data[2].sibling = 7
        self.link_data[2].child = 4
        self.link_data[2].mass = 0.029974
        self.link_data[2].relative_position = get_transition_xyz(-0.005, 0.0785, 0.148)
        self.link_data[2].joint_axis = get_transition_xyz(0.0, 1.0, 0.0)
        self.link_data[2].center_of_mass = get_transition_xyz(0, 0.01004, -0.005235)
        self.link_data[2].joint_limit_max = 0.75 * np.pi
        self.link_data[2].joint_limit_min = -0.75 * np.pi
        self.link_data[2].inertia = get_inertia_xyz(0.000009025, 0.0, 0.0, 0.000016881, 0.0, 0.000011316)
        
        # left arm shoulder roll
        self.link_data[4].name = "left_shoulder_roll_joint"
        self.link_data[4].parent = 2
        self.link_data[4].sibling = -1
        self.link_data[4].child = 6
        self.link_data[4].mass = 0.433169
        self.link_data[4].relative_position = get_transition_xyz(0, 0.0308, -0.025)
        self.link_data[4].joint_axis = get_transition_xyz(1.0, 0.0, 0.0)
        self.link_data[4].center_of_mass = get_transition_xyz(0.000415, 0.005221, -0.047591)
        self.link_data[4].joint_limit_max = np.pi
        self.link_data[4].joint_limit_min = 0.0
        self.link_data[4].inertia = get_inertia_xyz(0.001462386, 0.0, 0.0, 0.001573653, 0.0, 0.000250070)
        
        # left arm elbow
        self.link_data[6].name = "left_elbow_pitch_joint"
        self.link_data[6].parent = 4
        self.link_data[6].sibling = -1
        self.link_data[6].child = 22
        self.link_data[6].mass = 0.137614
        self.link_data[6].relative_position = get_transition_xyz(0, 0, -0.145)
        self.link_data[6].joint_axis = get_transition_xyz(0.0, 1.0, 0.0)
        self.link_data[6].center_of_mass = get_transition_xyz(0, 0.011654, -0.083221)
        self.link_data[6].joint_limit_max = 0.5 * np.pi
        self.link_data[6].joint_limit_min = -0.5 * np.pi
        self.link_data[6].inertia = get_inertia_xyz(0.000397774, 0.0, 0.0, 0.000391749, 0.0, 0.000056957)

        # left arm end effector
        self.link_data[22].name = "left_arm_end_joint"
        self.link_data[22].parent = 6
        self.link_data[22].sibling = -1
        self.link_data[22].child = -1
        self.link_data[22].mass = 0.0
        self.link_data[22].relative_position = get_transition_xyz(0.0, 0.02375, -0.182)
        self.link_data[22].joint_axis = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[22].center_of_mass = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[22].joint_limit_max = 100.0
        self.link_data[22].joint_limit_min = -100.0
        self.link_data[22].inertia = get_inertia_xyz(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
    def _initialize_right_leg(self):
        """Initialize right leg links"""
        # right leg hip yaw
        self.link_data[7].name = "right_waist_yaw_joint"
        self.link_data[7].parent = 0
        self.link_data[7].sibling = 8
        self.link_data[7].child = 9
        self.link_data[7].mass = 0.03478
        self.link_data[7].relative_position = get_transition_xyz(0.0, -0.044, -0.015)
        self.link_data[7].joint_axis = get_transition_xyz(0.0, 0.0, 1.0)
        self.link_data[7].center_of_mass = get_transition_xyz(-0.008797, 0, -0.009724)
        self.link_data[7].joint_limit_max = 0.75 * np.pi
        self.link_data[7].joint_limit_min = -0.75 * np.pi
        self.link_data[7].inertia = get_inertia_xyz(0.000008114, 0.00000, 0.00000, 0.000023034, 0.00000, 0.000019287)
        
        # right leg hip roll
        self.link_data[9].name = "right_waist_roll_joint"
        self.link_data[9].parent = 7
        self.link_data[9].sibling = -1
        self.link_data[9].child = 11
        self.link_data[9].mass = 0.129389
        self.link_data[9].relative_position = get_transition_xyz(0.0, 0.0, -0.0475)
        self.link_data[9].joint_axis = get_transition_xyz(1.0, 0.0, 0.0)
        self.link_data[9].center_of_mass = get_transition_xyz(-0.011068, 0.000004, -0.000154)
        self.link_data[9].joint_limit_max = 0.75 * np.pi
        self.link_data[9].joint_limit_min = -0.75 * np.pi
        self.link_data[9].inertia = get_inertia_xyz(0.000055580, 0.00000, 0.00000, 0.000061752, 0.00000, 0.000079989)
        
        # right leg hip pitch
        self.link_data[11].name = "right_upper_knee_pitch_joint"
        self.link_data[11].parent = 9
        self.link_data[11].sibling = -1
        self.link_data[11].child = 13
        self.link_data[11].mass = 0.19127
        self.link_data[11].relative_position = get_transition_xyz(0.0, 0.0, -0.1)
        self.link_data[11].joint_axis = get_transition_xyz(0.0, 1.0, 0.0)
        self.link_data[11].center_of_mass = get_transition_xyz(-0.012553, 0.000997, -0.023840)
        self.link_data[11].joint_limit_max = 0.5 * np.pi
        self.link_data[11].joint_limit_min = -0.15 * np.pi
        self.link_data[11].inertia = get_inertia_xyz(0.000164790, 0.00000, 0.00000, 0.000151986, 0.00000, 0.000098349)
        
        # right leg knee
        self.link_data[13].name = "right_knee_pitch_joint"
        self.link_data[13].parent = 11
        self.link_data[13].sibling = -1
        self.link_data[13].child = 15
        self.link_data[13].mass = 0.048967
        self.link_data[13].relative_position = get_transition_xyz(0.0, 0.0, -0.057)
        self.link_data[13].joint_axis = get_transition_xyz(0.0, 1.0, 0.0)
        self.link_data[13].center_of_mass = get_transition_xyz(0.011314, 0.000000, -0.037777)
        self.link_data[13].joint_limit_max = 0.5 * np.pi
        self.link_data[13].joint_limit_min = -0.15 * np.pi
        self.link_data[13].inertia = get_inertia_xyz(0.000082632, 0.00000, 0.00000, 0.000049273, 0.00000, 0.000045494)
        
        # right leg ankle pitch
        self.link_data[15].name = "right_ankle_pitch_joint"
        self.link_data[15].parent = 13
        self.link_data[15].sibling = -1
        self.link_data[15].child = 17
        self.link_data[15].mass = 0.170076
        self.link_data[15].relative_position = get_transition_xyz(0, 0, -0.1)
        self.link_data[15].joint_axis = get_transition_xyz(0.0, 1.0, 0.0)
        self.link_data[15].center_of_mass = get_transition_xyz(-0.031024, 0.000563, 0.005073)
        self.link_data[15].joint_limit_max = 0.75 * np.pi
        self.link_data[15].joint_limit_min = -0.75 * np.pi
        self.link_data[15].inertia = get_inertia_xyz(0.000049799, 0.00000, 0.00000, 0.000213199, 0.00000, 0.000223639)
        
        # right leg ankle roll
        self.link_data[17].name = "right_ankle_roll_joint"
        self.link_data[17].parent = 15
        self.link_data[17].sibling = -1
        self.link_data[17].child = 31
        self.link_data[17].mass = 0.084354
        self.link_data[17].relative_position = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[17].joint_axis = get_transition_xyz(1.0, 0.0, 0.0)
        self.link_data[17].center_of_mass = get_transition_xyz(-0.009915, -0.014839, -0.038901)
        self.link_data[17].joint_limit_max = 0.5 * np.pi
        self.link_data[17].joint_limit_min = -0.5 * np.pi
        self.link_data[17].inertia = get_inertia_xyz(0.000072411, 0.00000, 0.00000, 0.000303046, 0.00000, 0.000358794)
        
        # right leg end (roll - to end foot with spacer is 52.8mm)
        self.link_data[31].name = "right_foot_end_joint"
        self.link_data[31].parent = 17
        self.link_data[31].sibling = -1
        self.link_data[31].child = 33
        self.link_data[31].mass = 0.0
        self.link_data[31].relative_position = get_transition_xyz(0.0, 0.0, -0.0535)
        self.link_data[31].joint_axis = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[31].center_of_mass = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[31].joint_limit_max = 100.0
        self.link_data[31].joint_limit_min = -100.0
        self.link_data[31].inertia = get_inertia_xyz(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # right leg toe
        self.link_data[33].name = "right_foot_toe_joint"
        self.link_data[33].parent = 31
        self.link_data[33].sibling = 35
        self.link_data[33].child = -1
        self.link_data[33].mass = 0.0
        self.link_data[33].relative_position = get_transition_xyz(0.070865, 0.0, 0.0)
        self.link_data[33].joint_axis = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[33].center_of_mass = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[33].joint_limit_max = 100.0
        self.link_data[33].joint_limit_min = -100.0
        self.link_data[33].inertia = get_inertia_xyz(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # right leg heel end
        self.link_data[35].name = "right_foot_heel_joint"
        self.link_data[35].parent = 31
        self.link_data[35].sibling = -1
        self.link_data[35].child = -1
        self.link_data[35].mass = 0.0
        self.link_data[35].relative_position = get_transition_xyz(-0.070865, 0.0, 0.0)
        self.link_data[35].joint_axis = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[35].center_of_mass = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[35].joint_limit_max = 100.0
        self.link_data[35].joint_limit_min = -100.0
        self.link_data[35].inertia = get_inertia_xyz(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _initialize_left_leg(self):
        """Initialize left leg links"""
        # left leg hip yaw
        self.link_data[8].name = "left_waist_yaw_joint"
        self.link_data[8].parent = 0
        self.link_data[8].sibling = -1
        self.link_data[8].child = 10
        self.link_data[8].mass = 0.03478
        self.link_data[8].relative_position = get_transition_xyz(0.0, 0.044, -0.015)
        self.link_data[8].joint_axis = get_transition_xyz(0.0, 0.0, 1.0)
        self.link_data[8].center_of_mass = get_transition_xyz(-0.008797, 0, -0.009724)
        self.link_data[8].joint_limit_max = 0.75 * np.pi
        self.link_data[8].joint_limit_min = -0.75 * np.pi
        self.link_data[8].inertia = get_inertia_xyz(0.000008114, 0.00000, 0.00000, 0.000023034, 0.00000, 0.000019287)
        
        # left leg hip roll
        self.link_data[10].name = "left_waist_roll_joint"
        self.link_data[10].parent = 8
        self.link_data[10].sibling = -1
        self.link_data[10].child = 12
        self.link_data[10].mass = 0.129389
        self.link_data[10].relative_position = get_transition_xyz(0.0, 0.0, -0.0475)
        self.link_data[10].joint_axis = get_transition_xyz(1.0, 0.0, 0.0)
        self.link_data[10].center_of_mass = get_transition_xyz(-0.011068, -0.000004, -0.000154)
        self.link_data[10].joint_limit_max = 0.25 * np.pi
        self.link_data[10].joint_limit_min = -0.25 * np.pi
        self.link_data[10].inertia = get_inertia_xyz(0.000055580, 0.00000, 0.00000, 0.000061752, 0.00000, 0.000079989)

        # left leg upper knee pitch
        self.link_data[12].name = "left_upper_knee_pitch_joint"
        self.link_data[12].parent = 10
        self.link_data[12].sibling = -1
        self.link_data[12].child = 14
        self.link_data[12].mass = 0.19127
        self.link_data[12].relative_position = get_transition_xyz(0, 0, -0.1)
        self.link_data[12].joint_axis = get_transition_xyz(0.0, 1.0, 0.0)
        self.link_data[12].center_of_mass = get_transition_xyz(-0.012553, -0.000997, -0.023840)
        self.link_data[12].joint_limit_max = 0.5 * np.pi
        self.link_data[12].joint_limit_min = -0.15 * np.pi
        self.link_data[12].inertia = get_inertia_xyz(0.000164790, 0.00000, 0.00000, 0.000151986, 0.00000, 0.000098349)

        # left leg knee pitch
        self.link_data[14].name = "left_knee_pitch_joint"
        self.link_data[14].parent = 12
        self.link_data[14].sibling = -1
        self.link_data[14].child = 16
        self.link_data[14].mass = 0.19127
        self.link_data[14].relative_position = get_transition_xyz(0.0, 0, -0.057000)
        self.link_data[14].joint_axis = get_transition_xyz(0.0, 1.0, 0.0)
        self.link_data[14].center_of_mass = get_transition_xyz(-0.012553, -0.000997, -0.023840)
        self.link_data[14].joint_limit_max = 0.5 * np.pi
        self.link_data[14].joint_limit_min = -0.15 * np.pi
        self.link_data[14].inertia = get_inertia_xyz(0.000164790, 0.00000, 0.00000, 0.000151986, 0.00000, 0.000098349)

        # left leg ankle pitch
        self.link_data[16].name = "left_ankle_pitch_joint"
        self.link_data[16].parent = 14
        self.link_data[16].sibling = -1
        self.link_data[16].child = 18
        self.link_data[16].mass = 0.170076
        self.link_data[16].relative_position = get_transition_xyz(0.0, 0, -0.1)
        self.link_data[16].joint_axis = get_transition_xyz(0.0, 1.0, 0.0)
        self.link_data[16].center_of_mass = get_transition_xyz(-0.031024, -0.000563, 0.005073)
        self.link_data[16].joint_limit_max = 0.75 * np.pi
        self.link_data[16].joint_limit_min = -0.75 * np.pi
        self.link_data[16].inertia = get_inertia_xyz(0.000049799, 0.00000, 0.00000, 0.000213199, 0.00000, 0.000223639)
        
        # left leg ankle roll
        self.link_data[18].name = "left_ankle_roll_joint"
        self.link_data[18].parent = 16
        self.link_data[18].sibling = -1
        self.link_data[18].child = 32
        self.link_data[18].mass = 0.084354
        self.link_data[18].relative_position = get_transition_xyz(0.000, 0.000, 0.0)
        self.link_data[18].joint_axis = get_transition_xyz(1.0, 0.0, 0.0)
        self.link_data[18].center_of_mass = get_transition_xyz(-0.009915, 0.014839, -0.038901)
        self.link_data[18].joint_limit_max = 0.5 * np.pi
        self.link_data[18].joint_limit_min = -0.5 * np.pi
        self.link_data[18].inertia = get_inertia_xyz(0.000072411, 0.00000, 0.00000, 0.000303046, 0.00000, 0.000358794)

        # left leg end (roll - to end foot with spacer is 52.8mm)
        self.link_data[32].name = "left_foot_end_joint"
        self.link_data[32].parent = 18
        self.link_data[32].sibling = -1
        self.link_data[32].child = 34
        self.link_data[32].mass = 0.0
        self.link_data[32].relative_position = get_transition_xyz(0.0, 0.0, -0.0535)
        self.link_data[32].joint_axis = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[32].center_of_mass = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[32].joint_limit_max = 100.0
        self.link_data[32].joint_limit_min = -100.0
        self.link_data[32].inertia = get_inertia_xyz(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # left leg toe
        self.link_data[34].name = "left_foot_toe_joint"
        self.link_data[34].parent = 32
        self.link_data[34].sibling = 36
        self.link_data[34].child = -1
        self.link_data[34].mass = 0.0
        self.link_data[34].relative_position = get_transition_xyz(0.070865, 0.0, 0.0)
        self.link_data[34].joint_axis = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[34].center_of_mass = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[34].joint_limit_max = 100.0
        self.link_data[34].joint_limit_min = -100.0
        self.link_data[34].inertia = get_inertia_xyz(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # left leg heel end
        self.link_data[36].name = "left_heel_joint"
        self.link_data[36].parent = 32
        self.link_data[36].sibling = -1
        self.link_data[36].child = -1
        self.link_data[36].mass = 0.0
        self.link_data[36].relative_position = get_transition_xyz(-0.070865, 0.0, 0.0)
        self.link_data[36].joint_axis = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[36].center_of_mass = get_transition_xyz(0.0, 0.0, 0.0)
        self.link_data[36].joint_limit_max = 100.0
        self.link_data[36].joint_limit_min = -100.0
        self.link_data[36].inertia = get_inertia_xyz(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def find_route(self, to: int) -> List[int]:
        """Find route from base to target joint
        
        Args:
            to: Target joint ID
            
        Returns:
            List of joint IDs in the route
        """
        id = self.link_data[to].parent
        idx = []
        
        if id == 0:
            idx.append(0)
            idx.append(to)
        else:
            idx = self.find_route(id)
            idx.append(to)
            
        return idx
        
    def find_route_from_to(self, from_id: int, to: int) -> List[int]:
        """Find route from source to target joint
        
        Args:
            from_id: Source joint ID
            to: Target joint ID
            
        Returns:
            List of joint IDs in the route
        """
        id = self.link_data[to].parent
        idx = []
        
        if id == from_id:
            idx.append(from_id)
            idx.append(to)
        elif id != 0:
            idx = self.find_route_from_to(from_id, id)
            idx.append(to)
            
        return idx
        
    def calc_total_mass(self, joint_id: int) -> float:
        """Calculate total mass of a joint and its children
        
        Args:
            joint_id: Joint ID
            
        Returns:
            Total mass
        """
        if joint_id == -1:
            return 0.0
        
        mass = self.link_data[joint_id].mass
        mass += self.calc_total_mass(self.link_data[joint_id].sibling)
        mass += self.calc_total_mass(self.link_data[joint_id].child)
        
        return mass
        
    def calc_mc(self, joint_id: int) -> np.ndarray:
        """Calculate mass * center of mass for a joint and its children
        
        Args:
            joint_id: Joint ID
            
        Returns:
            Mass * center of mass
        """
        if joint_id == -1:
            return np.zeros((3, 1))
        
        mc = self.link_data[joint_id].mass * (
            self.link_data[joint_id].rotation @ self.link_data[joint_id].center_of_mass.reshape(3, 1) + 
            self.link_data[joint_id].position.reshape(3, 1)
        )
        
        mc = mc + self.calc_mc(self.link_data[joint_id].sibling) + self.calc_mc(self.link_data[joint_id].child)
        
        return mc
        
    def calc_com(self, mc: np.ndarray) -> np.ndarray:
        """Calculate center of mass
        
        Args:
            mc: Mass * center of mass
            
        Returns:
            Center of mass
        """
        mass = self.calc_total_mass(0)
        return mc / mass
        
    def calc_forward_kinematics(self, joint_id: int) -> None:
        """Calculate forward kinematics
        
        Args:
            joint_id: Joint ID
        """
        if joint_id == -1:
            return
            
        if joint_id == 0:
            # self.link_data[0].position = np.zeros((3, 1))
            # Use the imported calc_rodrigues and calc_hatto functions
            axis_hatto = calc_hatto(self.link_data[0].joint_axis)
            self.link_data[0].rotation = calc_rodrigues(axis_hatto, self.link_data[0].joint_angle)
            
        if joint_id != 0:
            parent = self.link_data[joint_id].parent
            
            self.link_data[joint_id].position = (
                self.link_data[parent].rotation @ 
                self.link_data[joint_id].relative_position.reshape(3, 1) + 
                self.link_data[parent].position.reshape(3, 1)
            ).reshape(3, 1)
            
            # Use the imported calc_rodrigues and calc_hatto functions
            axis_hatto = calc_hatto(self.link_data[joint_id].joint_axis)
            rot_matrix = calc_rodrigues(axis_hatto, self.link_data[joint_id].joint_angle)
            self.link_data[joint_id].rotation = self.link_data[parent].rotation @ rot_matrix
            
            # Update transformation matrix
            self.link_data[joint_id].transformation = np.eye(4)
            self.link_data[joint_id].transformation[:3, 3:4] = self.link_data[joint_id].position
            self.link_data[joint_id].transformation[:3, :3] = self.link_data[joint_id].rotation
            
        # Recursively calculate forward kinematics for siblings and children
        self.calc_forward_kinematics(self.link_data[joint_id].sibling)
        self.calc_forward_kinematics(self.link_data[joint_id].child)
        
    def calc_jacobian(self, idx: List[int]) -> np.ndarray:
        """Calculate Jacobian matrix
        
        Args:
            idx: List of joint IDs in the route
            
        Returns:
            Jacobian matrix
        """
        idx_size = len(idx)
        end = idx_size - 1
        
        tar_position = self.link_data[idx[end]].position.reshape(3, 1)
        jacobian = np.zeros((6, idx_size))
        
        for id in range(idx_size):
            curr_id = idx[id]
            
            tar_rotation = self.link_data[curr_id].rotation @ self.link_data[curr_id].joint_axis.reshape(3, 1)

            # Calculate cross product and ensure it's a column vector for assignment
            cross_product = calc_cross(
                tar_rotation.flatten(),  # Ensure 1D array
                (tar_position - self.link_data[curr_id].position.reshape(3, 1)).flatten()  # Ensure consistent shapes
            )
            jacobian[:3, id] = cross_product  # Assign to column directly
            
            # Assign rotation vector to lower part of Jacobian
            jacobian[3:, id] = tar_rotation.flatten()  # Ensure 1D array
            
        return jacobian
        
    def calc_jacobian_com(self, idx: List[int]) -> np.ndarray:
        """Calculate Jacobian matrix for center of mass
        
        Args:
            idx: List of joint IDs in the route
            
        Returns:
            Jacobian matrix for center of mass
        """
        idx_size = len(idx)
        end = idx_size - 1
        
        # Center of mass position is used in calculations below
        com_position = self.get_center_of_mass()
        jacobian_com = np.zeros((6, idx_size))
        
        for id in range(idx_size):
            curr_id = idx[id]
            mass = self.calc_total_mass(curr_id)
            
            og = self.calc_mc(curr_id) / mass - self.link_data[curr_id].position
            tar_rotation = self.link_data[curr_id].rotation @ self.link_data[curr_id].joint_axis.reshape(3, 1)
            
            jacobian_com[:3, id:id+1] = calc_cross(tar_rotation, og)
            jacobian_com[3:, id:id+1] = tar_rotation
            
        return jacobian_com
        
    def calc_vw_err(self, tar_position: np.ndarray, curr_position: np.ndarray,
                   tar_rotation: np.ndarray, curr_rotation: np.ndarray) -> np.ndarray:
        """Calculate velocity and angular velocity error
        
        Args:
            tar_position: Target position
            curr_position: Current position
            tar_rotation: Target orientation
            curr_rotation: Current orientation
            
        Returns:
            Error vector
        """
        pos_err = tar_position - curr_position
        ori_err = curr_rotation.T @ tar_rotation
        ori_err_dash = curr_rotation @ convert_rot_to_omega(ori_err)
        
        err = np.zeros((6, 1))
        err[:3] = pos_err.reshape(3, 1)
        err[3:] = ori_err_dash.reshape(3, 1)
        
        return err
        
    def _calc_inverse_kinematics_core(self, to: int, tar_position: np.ndarray, tar_rotation: np.ndarray,
                                   idx: List[int], max_iter: int = 100, ik_err: float = 1e-3) -> bool:
        """Core inverse kinematics calculation function
        
        This is a helper function that implements the core inverse kinematics algorithm
        used by both calc_inverse_kinematics and calc_inverse_kinematics_from_to.
        
        Args:
            to: Target joint ID
            tar_position: Target position
            tar_rotation: Target orientation
            idx: List of joint IDs in the route
            max_iter: Maximum number of iterations
            ik_err: Error threshold
            
        Returns:
            True if successful, False otherwise
        """
        ik_success = False
        limit_success = False
        
        # Validate inputs
        if to not in self.link_data:
            logger.error(f"Error: Target joint ID {to} does not exist")
            return False
            
        if len(idx) == 0:
            print("Error: No joints in route")
            return False
        
        # Iterative IK algorithm
        for iter in range(max_iter):
            jacobian = self.calc_jacobian(idx)
            
            curr_position = self.link_data[to].position
            curr_rotation = self.link_data[to].rotation
            err = self.calc_vw_err(tar_position, curr_position, tar_rotation, curr_rotation)
            
            if np.linalg.norm(err) < ik_err:
                ik_success = True
                break
            else:
                ik_success = False
                
            # Calculate pseudo-inverse of Jacobian
            jacobian_trans = jacobian @ jacobian.T
            print(f"Got jacobian_trans {jacobian_trans.shape}\n:", jacobian_trans)
            try:
                jacobian_inverse = jacobian.T @ np.linalg.inv(jacobian_trans)
            except Exception as ex:
                logger.error("Error during Jacobian calculation: {ex}")
                break
            
            # Calculate joint angle update
            delta_angle = jacobian_inverse @ err
            
            # Update joint angles
            for id in range(len(idx)):
                joint_num = idx[id]
                self.link_data[joint_num].joint_angle += delta_angle[id, 0]
                
            # Update forward kinematics
            self.calc_forward_kinematics(0)
            
        # Check joint limits
        limit_success = True  # Assume success until a limit violation is found
        for id in range(len(idx)):
            joint_num = idx[id]
            
            if self.link_data[joint_num].joint_angle >= self.link_data[joint_num].joint_limit_max:
                limit_success = False
                print(f"Joint {joint_num} ({self.link_data[joint_num].name}) exceeds upper limit")
                break
            elif self.link_data[joint_num].joint_angle <= self.link_data[joint_num].joint_limit_min:
                limit_success = False
                print(f"Joint {joint_num} ({self.link_data[joint_num].name}) exceeds lower limit")
                break
                
        return ik_success and limit_success
        
    def calc_inverse_kinematics(self, to: int, tar_position: np.ndarray, tar_rotation: np.ndarray,
                               max_iter: int = 100, ik_err: float = 1e-3) -> bool:
        """Calculate inverse kinematics
        
        Args:
            to: Target joint ID
            tar_position: Target position
            tar_rotation: Target orientation
            max_iter: Maximum number of iterations
            ik_err: Error threshold
            
        Returns:
            True if successful, False otherwise
        """
        # Validate input
        if to not in self.link_data:
            logger.error(f"Error: Target joint ID {to} does not exist")
            return False
            
        idx = self.find_route(to)
        return self._calc_inverse_kinematics_core(to, tar_position, tar_rotation, idx, max_iter, ik_err)
        
    def calc_inverse_kinematics_from_to(self, from_id: int, to: int, tar_position: np.ndarray, tar_rotation: np.ndarray,
                                       max_iter: int = 100, ik_err: float = 1e-3) -> bool:
        """Calculate inverse kinematics from a specific joint
        
        Args:
            from_id: Source joint ID
            to: Target joint ID
            tar_position: Target position
            tar_rotation: Target orientation
            max_iter: Maximum number of iterations
            ik_err: Error threshold
            
        Returns:
            True if successful, False otherwise
        """
        # Validate inputs
        if from_id not in self.link_data:
            logger.error(f"Error: Source joint ID {from_id} does not exist")
            return False
            
        if to not in self.link_data:
            logger.error(f"Error: Target joint ID {to} does not exist")
            return False
            
        idx = self.find_route_from_to(from_id, to)
        return self._calc_inverse_kinematics_core(to, tar_position, tar_rotation, idx, max_iter, ik_err)

    def get_joint_id_by_name(self, joint_name: str) -> int:
        """Find joint ID by name
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            Joint ID if found, -1 otherwise
        """
        for joint_id, link in self.link_data.items():
            if hasattr(link, 'name') and link.name == joint_name:
                return joint_id
        return -1
        
    def calc_analytical_inverse_kinematics_left_leg(self, tar_position: np.ndarray, tar_rotation: np.ndarray) -> bool:
        """Calculate analytical inverse kinematics for the left leg and set joint angles directly
        
        Args:
            tar_position: Target position in body frame (x, y, z)
            tar_rotation: Target orientation in body frame (roll, pitch, yaw)
        
        Returns:
            True if all joint angles were set successfully, False otherwise
        """
        # Define link lengths (based on the original kinematics.py)
        L1 = 0.118  # Upper leg length
        L12 = 0.023  # Hip offset
        L2 = 0.118  # Lower leg length
        L3 = 0.043  # Ankle to foot offset
        OFFSET_X = 0.0  # X offset from body to hip
        OFFSET_W = 0.044  # Y offset from body to hip (half of hip width)
        
        # Extract target position and orientation
        l_x, l_y, l_z = tar_position
        l_roll, l_pitch, l_yaw = tar_rotation
        
        # Apply offsets
        l_x -= OFFSET_X
        l_y -= OFFSET_W
        l_z = L1 + L12 + L2 + L3 - l_z
        
        # Transform to leg coordinate system
        l_x2 = l_x * np.cos(l_yaw) + l_y * np.sin(l_yaw)
        l_y2 = -l_x * np.sin(l_yaw) + l_y * np.cos(l_yaw)
        l_z2 = l_z - L3
        
        # Calculate hip roll angle
        waist_roll = np.arctan2(l_y2, l_z2)
        
        # Calculate remaining leg parameters
        l2 = l_y2**2 + l_z2**2
        l_z3 = np.sqrt(max(l2 - l_x2**2, 0.0)) - L12
        pitch = np.arctan2(l_x2, l_z3)
        l = np.sqrt(l_x2**2 + l_z3**2)
        
        # Calculate knee angle using cosine law
        knee_disp = np.arccos(np.clip(l/(2.0*L1), -1.0, 1.0))
        
        # Calculate hip and ankle pitch angles
        waist_pitch = -pitch - knee_disp
        knee_pitch = -pitch + knee_disp
        
        # Set joint angles directly
        joint_angles = {
            "left_waist_yaw_joint": -l_yaw,
            "left_waist_roll_joint": waist_roll,
            "left_upper_knee_pitch_joint": -waist_pitch,
            "left_knee_pitch_joint": knee_pitch,
            "left_ankle_pitch_joint": -l_pitch,
            "left_ankle_roll_joint": l_roll - waist_roll
        }
        
        # Set all joint angles and check if successful
        success = True
        for joint_name, angle in joint_angles.items():
            joint_id = self.get_joint_id_by_name(joint_name)
            if joint_id != -1:
                if not self.set_joint_angle(joint_id, angle):
                    success = False
            else:
                logger.error(f"Joint {joint_name} not found")
                success = False
                
        # Update forward kinematics if all joints were set successfully
        # if success:
            # self.calc_forward_kinematics(0)
            
        return success

    def calc_analytical_inverse_kinematics_right_leg(self, tar_position: np.ndarray, tar_rotation: np.ndarray) -> bool:
        """Calculate analytical inverse kinematics for the right leg and set joint angles directly
        
        Args:
            tar_position: Target position in body frame (x, y, z)
            tar_rotation: Target orientation in body frame (roll, pitch, yaw)
        
        Returns:
            True if all joint angles were set successfully, False otherwise
        """
        # Define link lengths (based on the original kinematics.py)
        L1 = 0.118  # Upper leg length
        L12 = 0.023  # Hip offset
        L2 = 0.118  # Lower leg length
        L3 = 0.043  # Ankle to foot offset
        OFFSET_X = 0.0  # X offset from body to hip
        OFFSET_W = 0.044  # Y offset from body to hip (half of hip width)
        
        # Extract target position and orientation
        r_x, r_y, r_z = tar_position
        r_roll, r_pitch, r_yaw = tar_rotation
        
        # Apply offsets (note the sign change for Y offset compared to left leg)
        r_x -= OFFSET_X
        r_y += OFFSET_W  # Add offset for right leg
        r_z = L1 + L12 + L2 + L3 - r_z
        
        # Transform to leg coordinate system
        r_x2 = r_x * np.cos(r_yaw) + r_y * np.sin(r_yaw)
        r_y2 = -r_x * np.sin(r_yaw) + r_y * np.cos(r_yaw)
        r_z2 = r_z - L3
        
        # Calculate hip roll angle
        waist_roll = np.arctan2(r_y2, r_z2)
        
        # Calculate remaining leg parameters
        r2 = r_y2**2 + r_z2**2
        r_z3 = np.sqrt(max(r2 - r_x2**2, 0.0)) - L12
        pitch = np.arctan2(r_x2, r_z3)
        l = np.sqrt(r_x2**2 + r_z3**2)
        
        # Calculate knee angle using cosine law
        knee_disp = np.arccos(np.clip(l/(2.0*L1), -1.0, 1.0))
        
        # Calculate hip and ankle pitch angles
        waist_pitch = -pitch - knee_disp
        knee_pitch = -pitch + knee_disp
        
        # Set joint angles directly
        joint_angles = {
            "right_waist_yaw_joint": -r_yaw,
            "right_waist_roll_joint": waist_roll,
            "right_upper_knee_pitch_joint": -waist_pitch,
            "right_knee_pitch_joint": knee_pitch,
            "right_ankle_pitch_joint": -r_pitch,
            "right_ankle_roll_joint": r_roll - waist_roll
        }
        
        # Set all joint angles and check if successful
        success = True
        for joint_name, angle in joint_angles.items():
            joint_id = self.get_joint_id_by_name(joint_name)
            if joint_id != -1:
                if not self.set_joint_angle(joint_id, angle):
                    success = False
            else:
                logger.error(f"Joint {joint_name} not found")
                success = False
                
        # Update forward kinematics if all joints were set successfully
        # if success:
            # self.calc_forward_kinematics(0)
            
        return success

    def set_joint_angle(self, joint_id: int, angle: float) -> bool:
        """Set joint angle with limit checking
        
        Args:
            joint_id: Joint ID
            angle: Joint angle in radians
            
        Returns:
            True if successful, False if joint_id is invalid or angle exceeds limits
        """
        # Validate joint ID
        if joint_id not in self.link_data:
            logger.error(f"Error: Joint ID {joint_id} does not exist")
            return False
            
        # Check joint limits
        if angle > self.link_data[joint_id].joint_limit_max:
            print(f"Warning: Joint {joint_id} ({self.link_data[joint_id].name}) angle {angle:.4f} exceeds upper limit {self.link_data[joint_id].joint_limit_max:.4f}")
            return False
        elif angle < self.link_data[joint_id].joint_limit_min:
            print(f"Warning: Joint {joint_id} ({self.link_data[joint_id].name}) angle {angle:.4f} exceeds lower limit {self.link_data[joint_id].joint_limit_min:.4f}")
            return False
            
        # Set joint angle
        self.link_data[joint_id].joint_angle = angle
        return True
        
    def set_joint_angles(self, joint_ids: List[int], angles: List[float]) -> bool:
        """Set multiple joint angles with limit checking
        
        Args:
            joint_ids: List of joint IDs
            angles: List of joint angles in radians
            
        Returns:
            True if all angles were set successfully, False otherwise
        """
        # Check that the lists have the same length
        if len(joint_ids) != len(angles):
            logger.error(f"Error: Number of joint IDs ({len(joint_ids)}) does not match number of angles ({len(angles)})")
            return False
            
        success = True
        for i, joint_id in enumerate(joint_ids):
            if not self.set_joint_angle(joint_id, angles[i]):
                success = False
                
        return success
            
    def get_joint_angle(self, joint_id: int) -> float:
        """Get joint angle
        
        Args:
            joint_id: Joint ID
            
        Returns:
            Joint angle in radians or 0.0 if joint_id is invalid
        """
        if joint_id not in self.link_data:
            logger.error(f"Error: Joint ID {joint_id} does not exist")
            return 0.0
        return self.link_data[joint_id].joint_angle
        
    def get_joint_angles(self, joint_ids: List[int]) -> List[float]:
        """Get multiple joint angles
        
        Args:
            joint_ids: List of joint IDs
            
        Returns:
            List of joint angles in radians (returns 0.0 for invalid joint IDs)
        """
        angles = []
        for joint_id in joint_ids:
            if joint_id not in self.link_data:
                logger.error(f"Error: Joint ID {joint_id} does not exist")
                angles.append(0.0)
            else:
                angles.append(self.link_data[joint_id].joint_angle)
        return angles
        
    def get_position(self, joint_id: int) -> np.ndarray:
        """Get joint position
        
        Args:
            joint_id: Joint ID
            
        Returns:
            Joint position or zeros if joint_id is invalid
        """
        if joint_id not in self.link_data:
            logger.error(f"Error: Joint ID {joint_id} does not exist")
            return np.zeros(3)
        return self.link_data[joint_id].position

    def set_position(self, joint_id: int, position: np.ndarray) -> bool:
        """Set joint position
        
        Args:
            joint_id: Joint ID
            position: Joint position
            
        Returns:
            True if successful, False if joint_id is invalid
        """
        if joint_id not in self.link_data:
            logger.error(f"Error: Joint ID {joint_id} does not exist")
            return False
            
        if position.shape != (3,1):
            logger.error(f"Error: Position must be a 3D vector, got shape {position.shape}")
            return False
            
        self.link_data[joint_id].position = position.reshape(3, 1)
        return True
        
    def get_rotation(self, joint_id: int) -> np.ndarray:
        """Get joint rotation
        
        Args:
            joint_id: Joint ID
            
        Returns:
            Joint rotation or identity matrix if joint_id is invalid
        """
        if joint_id not in self.link_data:
            logger.error(f"Error: Joint ID {joint_id} does not exist")
            return np.eye(3)
            
        return self.link_data[joint_id].rotation

    def set_rotation(self, joint_id: int, rotation: np.ndarray) -> bool:
        """Set joint rotation
        
        Args:
            joint_id: Joint ID
            rotation: Joint rotation
            
        Returns:
            True if successful, False if joint_id is invalid or rotation is invalid
        """
        if joint_id not in self.link_data:
            logger.error(f"Error: Joint ID {joint_id} does not exist")
            return False
            
        if rotation.shape != (3, 3):
            logger.error(f"Error: Rotation must be a 3x3 matrix, got shape {rotation.shape}")
            return False
            
        self.link_data[joint_id].rotation = rotation
        return True

    def get_transformation(self, joint_id: int) -> np.ndarray:
        """Get joint transformation matrix
        
        Args:
            joint_id: Joint ID
            
        Returns:
            Joint transformation matrix or identity matrix if joint_id is invalid
        """
        if joint_id not in self.link_data:
            logger.error(f"Error: Joint ID {joint_id} does not exist")
            return np.eye(4)
            
        return self.link_data[joint_id].transformation

    def set_transformation(self, joint_id: int, transformation: np.ndarray) -> bool:
        """Set joint transformation matrix
        
        Args:
            joint_id: Joint ID
            transformation: Joint transformation matrix
            
        Returns:
            True if successful, False if joint_id is invalid or transformation is invalid
        """
        if joint_id not in self.link_data:
            logger.error(f"Error: Joint ID {joint_id} does not exist")
            return False
            
        if transformation.shape != (4, 4):
            logger.error(f"Error: Transformation must be a 4x4 matrix, got shape {transformation.shape}")
            return False
            
        self.link_data[joint_id].transformation = transformation
        self.link_data[joint_id].position = transformation[:3, 3].reshape(3,1)
        self.link_data[joint_id].rotation = transformation[:3, :3]
        return True

    def get_center_of_mass(self) -> np.ndarray:
        """Get center of mass of the entire robot
        
        Returns:
            Center of mass
        """
        mc = self.calc_mc(0)
        return self.calc_com(mc)
        
    def get_joint_axis(self, link_name: str) -> np.ndarray:
        """Get joint axis of a link by name
        
        Args:
            link_name: Name of the link
            
        Returns:
            Joint axis vector or zeros if link_name is invalid
        """
        joint_axis = np.zeros(3)
        
        # Find the link by name
        link_data = None
        for joint_id, link in self.link_data.items():
            if hasattr(link, 'name') and link.name == link_name:
                link_data = link
                break
                
        if link_data is not None:
            joint_axis = link_data.joint_axis
            
        return joint_axis
        
    def get_joint_direction(self, link_identifier) -> float:
        """Get joint direction as sum of axis components
        
        Args:
            link_identifier: Link ID (int) or link name (str)
            
        Returns:
            Joint direction (sum of joint axis components)
        """
        joint_direction = 0.0
        link_data = None
        
        # Handle both string and int identifiers
        if isinstance(link_identifier, str):
            # Find the link by name
            for joint_id, link in self.link_data.items():
                if hasattr(link, 'name') and link.name == link_identifier:
                    link_data = link
                    break
        elif isinstance(link_identifier, int):
            # Get link by ID
            if link_identifier in self.link_data:
                link_data = self.link_data[link_identifier]
        
        if link_data is not None:
            # Sum the components of the joint axis
            joint_direction = link_data.joint_axis[0] + link_data.joint_axis[1] + link_data.joint_axis[2]
            
        return joint_direction
