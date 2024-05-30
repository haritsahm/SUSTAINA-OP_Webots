#!/usr/bin/env python3
# 
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
# Copyright 2024 Yasuo Hayashibara,Hiroki Noguchi
# generating walking pattern for the SUSTAINA-OP

import numpy as np
from kinematics import *
from foot_step_planner import *
from preview_control import *
from time import sleep
import csv

class walking():
  def __init__(self, dt, motorNames, left_foot0, right_foot0, joint_angles, pc):
    self.dt = dt
    self.kine = kinematics(motorNames)
    self.left_foot0, self.right_foot0 = left_foot0, right_foot0
    self.joint_angles = joint_angles
    self.pc = pc
    self.period = 0.32
    self.fsp = foot_step_planner(0.10, 0.04, 0.4, self.period, 0.06)
    self.X = np.matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    self.pattern = []
    self.left_up = self.right_up = 0.0
    self.left_off,  self.left_off_g,  self.left_off_d  = np.matrix([[0.0, 0.0, 0.0]]),  np.matrix([[0.0, 0.0, 0.0]]),  np.matrix([[0.0, 0.0, 0.0]]) 
    self.right_off, self.right_off_g, self.right_off_d = np.matrix([[0.0, 0.0, 0.0]]),  np.matrix([[0.0, 0.0, 0.0]]),  np.matrix([[0.0, 0.0, 0.0]])
    self.th = 0
    self.status = 'start'
    self.next_leg = 'right'
    self.foot_step = []
    return

  def setGoalPos(self, pos = None):
    if pos == None:
      if len(self.foot_step) <= 4:
        self.status = 'start'
      if len(self.foot_step) > 3:
        del self.foot_step[0]
    else:
      if len(self.foot_step) > 2:
        if not self.status == 'start':
          offset_y = -0.06 if self.next_leg == 'left' else 0.06
        else:
          offset_y = 0.0
        current_x, current_y, current_th = self.foot_step[1][1], self.foot_step[1][2]+offset_y, self.foot_step[1][3]
      else:
        current_x, current_y, current_th = 0, 0, 0
      self.foot_step = self.fsp.calculate(pos[0], pos[1], pos[2], current_x, current_y, current_th, self.next_leg, self.status)
      self.status = 'walking'
#    print(str(self.foot_step)+'\n')
    t = self.foot_step[0][0]
    self.pattern, x, y = self.pc.set_param(t, self.X[:,0], self.X[:,1], self.foot_step)
    self.X = np.matrix([[x[0,0], y[0,0]], [x[1,0], y[1,0]], [x[2,0], y[2,0]]])
    if self.foot_step[0][4] == 'left':
      if  self.foot_step[1][4] == 'both':
        self.right_off_g = np.matrix([[self.foot_step[1][1], self.foot_step[1][2], self.foot_step[1][3]]])
      else:
        self.right_off_g = np.matrix([[self.foot_step[1][1], self.foot_step[1][2]+0.06, self.foot_step[1][3]]])
      self.right_off_d = (self.right_off_g - self.right_off)/17.0
      self.next_leg = 'right'
    if self.foot_step[0][4] == 'right':
      if  self.foot_step[1][4] == 'both':
        self.left_off_g  = np.matrix([[self.foot_step[1][1], self.foot_step[1][2], self.foot_step[1][3]]])
      else:
        self.left_off_g  = np.matrix([[self.foot_step[1][1], self.foot_step[1][2]-0.06, self.foot_step[1][3]]])
      self.left_off_d  = (self.left_off_g - self.left_off)/17.0
      self.next_leg = 'left'
    self.th = self.foot_step[0][3]
    return self.foot_step

  def getNextPos(self):
    X = self.pattern.pop(0)
    period = round((self.foot_step[1][0]-self.foot_step[0][0])/self.dt)
    self.th += (self.foot_step[1][3]-self.foot_step[0][3])/period
    x_dir = 0
    BOTH_FOOT = round(self.period/2/self.dt)
    start_up = round(BOTH_FOOT/2)
    end_up   = round(period/2)
    period_up = end_up - start_up
    foot_hight = 0.10
    if self.foot_step[0][4] == 'right':
      # up or down foot
      if start_up < (period-len(self.pattern)) <= end_up:
        self.left_up  += foot_hight/period_up
      elif self.left_up > 0:
        self.left_up  = max(self.left_up  - foot_hight/period_up, 0.0)
      # move foot in the axes of x,y,the
      if (period-len(self.pattern)) > start_up:
        self.left_off += self.left_off_d
        if (period-len(self.pattern)) > (start_up + period_up * 2):
          self.left_off = self.left_off_g.copy()
    if self.foot_step[0][4] == 'left':
      # up or down foot
      if start_up < (period-len(self.pattern)) <= end_up:
        self.right_up += foot_hight/period_up
      elif self.right_up > 0:
        self.right_up = max(self.right_up - foot_hight/period_up, 0.0)
      # move foot in the axes of x,y,the
      if (period-len(self.pattern)) > start_up:
        self.right_off += self.right_off_d
        if (period-len(self.pattern)) > (start_up + period_up * 2):
          self.right_off = self.right_off_g.copy()
    lo = self.left_off  - np.block([[X[0,0:2],0]])
    ro = self.right_off - np.block([[X[0,0:2],0]])
    left_foot  = [self. left_foot0[0]+lo[0,0], self. left_foot0[1]+lo[0,1], self. left_foot0[2]+self.left_up , 0.0, 0.0, self.th-lo[0,2]]
    right_foot = [self.right_foot0[0]+ro[0,0], self.right_foot0[1]+ro[0,1], self.right_foot0[2]+self.right_up, 0.0, 0.0, self.th-ro[0,2]]
    self.joint_angles = self.kine.solve_ik(left_foot, right_foot, self.joint_angles)
    xp = [X[0,2], X[0,3]]

    return self.joint_angles, left_foot, right_foot, xp, len(self.pattern)