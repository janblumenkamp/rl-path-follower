from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
import os
import re
import scipy.signal
import yaml
import pybullet as p
import pybullet_data
import time
from pybullet_multicopter.copters.quadcopter import Quadcopter
from pybullet_multicopter.util.pid import PID
# Constants used for indexing.
X = 0
Y = 1
YAW = 2

CONFIG_SPACE_SIZE = 8
GOAL_POSITION = np.array([-1.5, 1.5], dtype=np.float32)

EPSILON = .1

#np.random.seed(2)

def sample_random_position():
    position = np.zeros(2, dtype=np.float32)
    position[X] = np.random.uniform(low=-CONFIG_SPACE_SIZE/2, high=CONFIG_SPACE_SIZE/2)
    position[Y] = np.random.uniform(low=-CONFIG_SPACE_SIZE/2, high=CONFIG_SPACE_SIZE/2)

    return position

def adjust_pose(node, final_position):
    final_pose = node.pose.copy()
    final_pose[:2] = final_position
    
    dp = final_pose[:2] - node.position
    beta = np.arctan2(dp[0], dp[1]) + node.yaw # gamma + alpha
    final_pose[YAW] = node.yaw + np.pi - 2*beta

    while final_pose[YAW] > np.pi:
        final_pose[YAW] -= 2*np.pi
    while final_pose[YAW] < -np.pi:
        final_pose[YAW] += 2*np.pi

    return Node(final_pose)

# Defines a node of the graph.
class Node(object):
    def __init__(self, pose):
        self._pose = pose.copy()
        self._neighbors = []
        self._parent = None

    @property
    def pose(self):
        return self._pose

    def add_neighbor(self, node):
        self._neighbors.append(node)

    def remove_neighbor(self, node):
        self._neighbors.remove(node)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, node):
        self._parent = node

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def position(self):
        return self._pose[:2]

    @property
    def yaw(self):
        return self._pose[YAW]
    
    @property
    def direction(self):
        return np.array([np.cos(self._pose[YAW]), np.sin(self._pose[YAW])], dtype=np.float32)

def rrt(start_pose, goal_position):
    # RRT builds a graph one node at a time.
    graph = []
    start_node = Node(start_pose)
    final_node = None
    graph.append(start_node)
    while True:
        position = sample_random_position()
        # With a random chance, draw the goal position.
        if np.random.rand() < .05:
            position = goal_position
        # Find closest node in graph.
        # In practice, one uses an efficient spatial structure (e.g., quadtree).
        potential_parent = sorted(((n, np.linalg.norm(position - n.position)) for n in graph), key=lambda x: x[1])
        # Pick a node at least some distance away but not too far.
        # We also verify that the angles are aligned (within pi / 4).
        u = None
        for n, d in potential_parent:
            if d > .2 and d < 1.5 and n.direction.dot(position - n.position) / d > 0.70710678118:
                u = n
                break
        else:
            continue
        v = adjust_pose(u, position)
        if v is None:
            continue
        u.add_neighbor(v)
        v.parent = u
        graph.append(v)
        if np.linalg.norm(v.position - goal_position) < .1:
            final_node = v
            break

    return start_node, final_node
  
def find_circle(node_a, node_b):
    def perpendicular(v):
        w = np.empty_like(v)
        w[X] = -v[Y]
        w[Y] = v[X]
        return w
    db = perpendicular(node_b.direction)
    dp = node_a.position - node_b.position
    t = np.dot(node_a.direction, db)
    if np.abs(t) < 1e-3:
        # By construction node_a and node_b should be far enough apart,
        # so they must be on opposite end of the circle.
        center = (node_b.position + node_a.position) / 2.
        radius = np.linalg.norm(center - node_b.position)
    else:
        radius = np.dot(node_a.direction, dp) / t
        center = radius * db + node_b.position
    return center, np.abs(radius)

def get_path(final_node, start_height, end_height):
  # Construct path from RRT solution.
  if final_node is None:
    return []
  path_reversed = []
  path_reversed.append(final_node)
  while path_reversed[-1].parent is not None:
    path_reversed.append(path_reversed[-1].parent)
  path = list(reversed(path_reversed))
  # Put a point every 5 cm.
  distance = 0.1
  offset = 0.
  points_x = []
  points_y = []
  for u, v in zip(path, path[1:]):
    center, radius = find_circle(u, v)
    du = u.position - center
    theta1 = np.arctan2(du[1], du[0])
    dv = v.position - center
    theta2 = np.arctan2(dv[1], dv[0])
    # Check if the arc goes clockwise.
    clockwise = np.cross(u.direction, du).item() > 0.
    # Generate a point every 5cm apart.
    da = distance / radius
    offset_a = offset / radius
    if clockwise:
      da = -da
      offset_a = -offset_a
      if theta2 > theta1:
        theta2 -= 2. * np.pi
    else:
      if theta2 < theta1:
        theta2 += 2. * np.pi
    angles = np.arange(theta1 + offset_a, theta2, da)
    offset = distance - (theta2 - angles[-1]) * radius
    points_x.extend(center[X] + np.cos(angles) * radius)
    points_y.extend(center[Y] + np.sin(angles) * radius)
  points_z = np.linspace(start_height, end_height, len(points_x))
  return np.stack((points_x, points_y, points_z), axis=-1)

def feedback_linearized(pose, velocity, epsilon):
  u = velocity[X]*np.cos(pose[YAW]) + velocity[Y]*np.sin(pose[YAW])  # [m/s]
  w = (1/epsilon)*(-velocity[X]*np.sin(pose[YAW]) + velocity[Y]*np.cos(pose[YAW]))  # [rad/s] going counter-clockwise.
  h = velocity[2]
  return u, w, h


def get_velocity(position, path_points):
  v = np.zeros_like(position)
  if len(path_points) == 0:
    return v
  # Stop moving if the goal is reached.
  if np.linalg.norm(position - path_points[-1]) < .2:
    return v

  closest_point_index = np.argmin(np.sum((path_points - position)**2, axis=1))
  next_destination = path_points[closest_point_index+3 if closest_point_index+3 < len(path_points) else len(path_points)-1]
  kappa=3
  v = (next_destination - position)*kappa
  return v

if __name__ == '__main__':
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -10, physicsClientId=client) 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf", physicsClientId=client)
    drone = Quadcopter(client)
    
    def get_drone_pose():
        return np.array(list(drone.position[:2])+[drone.orientation_euler[2]])

    while True:
        next_goal = np.random.uniform(-CONFIG_SPACE_SIZE/2, CONFIG_SPACE_SIZE/2, 3) + np.array([0, 0, CONFIG_SPACE_SIZE/2])
        start_node, final_node = rrt(get_drone_pose(), next_goal[:2])
        
        path = get_path(final_node, drone.position[2], next_goal[2])
        for i in range(1, len(path)):
            p.addUserDebugLine(path[i-1], path[i], lineColorRGB=[1,0,0], lineWidth=3, physicsClientId=client)
        while True:
            position = np.array([
                drone.position[X] + EPSILON * np.cos(drone.orientation_euler[YAW]),
                drone.position[Y] + EPSILON * np.sin(drone.orientation_euler[YAW]),
                drone.position[2]], dtype=np.float32)
            v = get_velocity(position, np.array(path, dtype=np.float32))
            u, w, h = feedback_linearized(get_drone_pose(), v, epsilon=EPSILON)
        
            drone.step_speed(u, 0, h)
            drone.set_yaw(w)
            drone.step()
            p.resetDebugVisualizerCamera(5, 270+drone.orientation_euler[2]*(180/np.pi), -35, drone.position, client)
            p.stepSimulation(physicsClientId=client)
            time.sleep(1/240)
            
            if np.linalg.norm(position-path[-1], ord=2) < 0.2:
                break
