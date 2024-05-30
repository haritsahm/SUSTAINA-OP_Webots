#!/usr/bin/env python3

# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
# Copyright 2024 Yasuo Hayashibara,Hiroki Noguchi,Satoshi Inoue

from controller import Robot
from kinematics import *
from foot_step_planner import *
from preview_control import *
from walking import *
from random import random 
import walk_server
import time

motorNames = [
  "head_yaw_joint",                        # ID1
  "left_shoulder_pitch_joint",             # ID2
  "left_shoulder_roll_joint",              # ID3
  "left_elbow_pitch_joint",                # ID4
  "right_shoulder_pitch_joint",            # ID5
  "right_shoulder_roll_joint",             # ID6
  "right_elbow_pitch_joint",               # ID7
  "left_waist_yaw_joint",                  # ID8
  "left_waist_roll_joint",                 # ID9
  "left_upper_knee_pitch_joint",           # ID10
  "left_knee_pitch_joint",                 # ID11
  "left_ankle_pitch_joint",                # ID12
  "left_ankle_roll_joint",                 # ID13
  "right_waist_yaw_joint",                 # ID14
  "right_waist_roll_joint",                # ID15
  "right_upper_knee_pitch_joint",          # ID16
  "right_knee_pitch_joint",                # ID17
  "right_ankle_pitch_joint",               # ID18
  "right_ankle_roll_joint"                 # ID19
]

if __name__ == '__main__':
  robot = Robot()
  timestep = int(robot.getBasicTimeStep())

  motor = [None]*len(motorNames)
  for i in range(len(motorNames)):
    motor[i] = robot.getDevice(motorNames[i])

  joint_angles = [0]*len(motorNames)

  left_foot  = [-0.02, +0.054, 0.02]
  right_foot = [-0.02, -0.054, 0.02]

  pc = preview_control(timestep/1000, 1.0, 0.27)
  walk = walking(timestep/1000, motorNames, left_foot, right_foot, joint_angles, pc)
  walk_server = walk_server.WalkServer()
  camera = robot.getDevice("camera_sensor")
  camera.enable(int(robot.getBasicTimeStep()))
  time.sleep(2)
  camera_fps = 75
  loop_counter = 0

  #goal position (x, y) theta
  # foot_step = walk.setGoalPos([0.4, 0.0, 0.5])
  foot_step = walk.setGoalPos([0.1, 0.0, 0.1])
  command = None
  while robot.step(timestep) != -1:
    # receive command
    if command is None:
      command = walk_server.getCommand()
      if command is not None:
        print("Received command...")
        print(command)

    # send camera image
    loop_counter += 1
    if (loop_counter * timestep) > (1000 / camera_fps):
      image = camera.getImage()
      width = camera.getWidth()
      height = camera.getHeight()
      walk_server.sendImageData(width, height, image)
      loop_counter = 0
    
    joint_angles,lf,rf,xp,n = walk.getNextPos()
    if n == 0:
      if command is not None:
        x_goal, y_goal, th = command.target_x, command.target_y, command.target_theta
        print("Goal: ("+str(x_goal)+", "+str(y_goal)+", "+str(th)+")")
        foot_step = walk.setGoalPos([x_goal, y_goal, th])
        command = None
      else:
        foot_step = walk.setGoalPos()
    for i in range(len(motorNames)):
      motor[i].setPosition(joint_angles[i])
    pass
