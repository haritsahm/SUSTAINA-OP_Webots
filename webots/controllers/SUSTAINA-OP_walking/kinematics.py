#!/usr/bin/env python3

# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
# Copyright 2024 Yasuo Hayashibara,Hiroki Noguchi
# solving kinematics for the SUSTAINA-OP

import math

class kinematics():
  L1  = 0.118
  L12 = 0.023
  L2  = 0.118
  L3  = 0.043
  OFFSET_W   = 0.044
  OFFSET_X   = -0.0275
  def __init__(self, motorNames):
    self.motorNames = motorNames

  def solve_ik(self, left_foot, right_foot, current_angles):
    joint_angles = current_angles.copy()
    l_x, l_y, l_z, l_roll, l_pitch, l_yaw = left_foot
    l_x -= self.OFFSET_X
    l_y -= self.OFFSET_W
    l_z = self.L1 + self.L12 + self.L2 + self.L3 - l_z
    l_x2 =  l_x * math.cos(l_yaw) + l_y * math.sin(l_yaw)
    l_y2 = -l_x * math.sin(l_yaw) + l_y * math.cos(l_yaw)
    l_z2 =  l_z - self.L3
    waist_roll = math.atan2(l_y2, l_z2)
    l2 = l_y2**2 + l_z2**2
    l_z3 = math.sqrt(max(l2 - l_x2**2, 0.0)) - self.L12
    pitch = math.atan2(l_x2, l_z3)
    l = math.sqrt(l_x2**2 + l_z3**2)
    knee_disp = math.acos(min(max(l/(2.0*self.L1),-1.0),1.0))
    waist_pitch = - pitch - knee_disp
    knee_pitch  = - pitch + knee_disp
    joint_angles[self.motorNames.index('left_waist_yaw_joint'       )] = -l_yaw
    joint_angles[self.motorNames.index('left_waist_roll_joint'      )] = waist_roll
    joint_angles[self.motorNames.index('left_upper_knee_pitch_joint'     )] = -waist_pitch
    joint_angles[self.motorNames.index('left_knee_pitch_joint'      )] = knee_pitch
    joint_angles[self.motorNames.index('left_ankle_pitch_joint'     )] = -l_pitch
    joint_angles[self.motorNames.index('left_ankle_roll_joint'      )] = l_roll - waist_roll

    r_x, r_y, r_z, r_roll, r_pitch, r_yaw = right_foot
    r_x -= self.OFFSET_X
    r_y += self.OFFSET_W
    r_z = self.L1 + self.L12 + self.L2 + self.L3 - r_z
    r_x2 =  r_x * math.cos(r_yaw) + r_y * math.sin(r_yaw)
    r_y2 = -r_x * math.sin(r_yaw) + r_y * math.cos(r_yaw)
    r_z2 =  r_z - self.L3
    waist_roll = math.atan2(r_y2, r_z2)
    r2 = r_y2**2 + r_z2**2
    r_z3 = math.sqrt(max(r2 - r_x2**2, 0.0)) - self.L12
    pitch = math.atan2(r_x2, r_z3)
    l = math.sqrt(r_x2**2 + r_z3**2)
    knee_disp = math.acos(min(max(l/(2.0*self.L1),-1.0),1.0))
    waist_pitch = - pitch - knee_disp
    knee_pitch  = - pitch + knee_disp
    joint_angles[self.motorNames.index('right_waist_yaw_joint'       )] = -r_yaw
    joint_angles[self.motorNames.index('right_waist_roll_joint'      )] = waist_roll
    joint_angles[self.motorNames.index('right_upper_knee_pitch_joint'     )] = -waist_pitch
    joint_angles[self.motorNames.index('right_knee_pitch_joint'      )] = knee_pitch
    joint_angles[self.motorNames.index('right_ankle_pitch_joint'     )] = -r_pitch
    joint_angles[self.motorNames.index('right_ankle_roll_joint'      )] = r_roll - waist_roll

    return joint_angles