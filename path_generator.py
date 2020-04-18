import numpy as np
#import pybullet as p
#import pybullet_data
import time
#from pybullet_multicopter.copters.quadcopter import Quadcopter

#np.random.seed(2)

# Constants used for indexing.
X = 0
Y = 1
Z = 2
YAW = 3

class Node(object):
    def __init__(self, pose):
        assert(len(pose.shape) == 1 and pose.shape[0] == 4)
        self._pose = pose.copy().astype(np.float)
        self._neighbors = []
        self._parent = None
        self._cost = 0

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
        return self._pose[:YAW]

    @property
    def position2D(self):
        return self._pose[:Z]

    @property
    def yaw(self):
        return self._pose[YAW]

    @property
    def direction(self):
        return np.array([np.cos(self._pose[YAW]), np.sin(self._pose[YAW])], dtype=np.float32)

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, c):
        self._cost = c

class PathGenerator():
    def __init__(self):
        pass

    def adjust_pose(self, node, final_position):
        final_pose = node.pose.copy().astype(np.float)
        final_pose[:YAW] = final_position

        dp = final_pose[:Z] - node.position[:Z]
        beta = np.arctan2(dp[X], dp[Y]) + node.yaw # gamma + alpha
        final_pose[YAW] = node.yaw + np.pi - 2*beta

        while final_pose[YAW] > np.pi:
            final_pose[YAW] -= 2*np.pi
        while final_pose[YAW] < -np.pi:
            final_pose[YAW] += 2*np.pi

        final_node = Node(final_pose)

        c, r = self.find_circle(node, final_node)
        def get_angle_to_node(nd):
            dp1 = nd.pose[:Z] - c
            return np.arctan2(dp1[Y], dp1[X])
        angle_node = get_angle_to_node(node)
        angle_final_node = get_angle_to_node(final_node)
        final_node.cost = node.cost + np.abs(angle_node - angle_final_node)*r
        return final_node

    def find_circle(self, node_a, node_b):
        def perpendicular(v):
            w = np.empty_like(v)
            w[X] = -v[Y]
            w[Y] = v[X]
            return w
        db = perpendicular(node_b.direction)
        dp = node_a.position2D - node_b.position2D
        t = np.dot(node_a.direction, db)
        if np.abs(t) < 1e-3:
            # By construction node_a and node_b should be far enough apart,
            # so they must be on opposite end of the circle.
            center = (node_b.position2D + node_a.position2D) / 2.
            radius = np.linalg.norm(center - node_b.position2D)
        else:
            radius = np.dot(node_a.direction, dp) / t
            center = radius * db + node_b.position2D
        return center, np.abs(radius)

    def _get_path(self, final_node):
        # Construct path from RRT solution.
        if final_node is None:
            return []
        path_reversed = []
        path_reversed.append(final_node)
        while path_reversed[-1].parent is not None:
            path_reversed.append(path_reversed[-1].parent)
        path = list(reversed(path_reversed))
        # Put a point every 5 cm.
        distance = 0.05
        offset = 0.
        points_x = []
        points_y = []
        points_z = []
        for u, v in zip(path, path[1:]):
            center, radius = self.find_circle(u, v)
            du = u.position2D - center
            theta1 = np.arctan2(du[1], du[0])
            dv = v.position2D - center
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
            if len(angles) > 0:
                offset = distance - (theta2 - angles[-1]) * radius
                points_x.extend(center[X] + np.cos(angles) * radius)
                points_y.extend(center[Y] + np.sin(angles) * radius)
                points_z.extend(np.linspace(u.position[Z], v.position[Z], len(angles)))
        return np.stack((points_x, points_y, points_z), axis=-1)

    def get_path(self, start_pose, min_length):
        start_node = Node(start_pose)
        final_node = None
        current_parent = start_node
        current_len = 1
        while True:
            position = np.array([
                np.random.uniform(low=current_parent.position[0] - 3, high=current_parent.position[0] + 3),
                np.random.uniform(low=current_parent.position[1] - 3, high=current_parent.position[1] + 3),
                max(0, np.random.uniform(low=max(current_parent.position[2] - 0.5, 0.5), high=current_parent.position[2] + 0.5))
            ])

            # We also verify that the angles are aligned (within pi / 4).
            d = np.linalg.norm(position - current_parent.position)
            if d > .2 and d < 1.5 and current_parent.direction.dot(position[:Z] - current_parent.position2D) / d > np.pi/8:
                v = self.adjust_pose(current_parent, position)
                if v is None:
                    continue
                current_parent.add_neighbor(v)
                v.parent = current_parent
                if current_len > min_length:
                    final_node = v
                    break
                current_len += 1
                current_parent = v

        return self._get_path(final_node)

def feedback_linearization(drone_pose, path, epsilon, kappa):
    def feedback_linearized(pose, velocity, epsilon):
        u = velocity[X]*np.cos(pose[YAW]) + velocity[Y]*np.sin(pose[YAW])  # [m/s]
        w = (1/epsilon)*(-velocity[X]*np.sin(pose[YAW]) + velocity[Y]*np.cos(pose[YAW]))  # [rad/s] going counter-clockwise.
        return u, w

    position = np.array([
        drone_pose[X] + EPSILON * np.cos(drone_pose[YAW]),
        drone_pose[Y] + EPSILON * np.sin(drone_pose[YAW]),
        drone_pose[Z]], dtype=np.float32)

    v = np.zeros_like(position)
    if len(path) > 0 and np.linalg.norm(position - path[-1]) > .2:
        closest_point_index = np.argmin(np.sum((path - position)**2, axis=1))
        next_destination = path[closest_point_index+1 if closest_point_index+1 < len(path) else len(path)-1]
        v = (next_destination - position)*kappa

    u, w = feedback_linearized(get_drone_pose(), v, epsilon=epsilon)
    h = v[Z]
    return u, w, h

'''
CONFIG_SPACE_SIZE = 8
EPSILON = 0.2
KAPPA = 3

if __name__ == '__main__':
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -10, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf", physicsClientId=client)
    drone = Quadcopter(client)

    def get_drone_pose():
        return np.array(list(drone.position)+[drone.orientation_euler[2]])

    path_generator = PathGenerator(np.array([[-CONFIG_SPACE_SIZE, CONFIG_SPACE_SIZE], [-CONFIG_SPACE_SIZE, CONFIG_SPACE_SIZE], [1, CONFIG_SPACE_SIZE]]))
    while True:
        path = path_generator.get_path(get_drone_pose(), path_generator.sample_configuration_space())
        for i in range(1, len(path)):
            p.addUserDebugLine(path[i-1], path[i], lineColorRGB=[1,0,0], lineWidth=3, physicsClientId=client)

        while True:
            u, w, h = feedback_linearization(get_drone_pose(), path, EPSILON, KAPPA)
            drone.step_speed(u, 0, h)
            drone.set_yaw(w)
            drone.step()
            p.resetDebugVisualizerCamera(5, 270+drone.orientation_euler[2]*(180/np.pi), -35, drone.position, client)
            p.stepSimulation(physicsClientId=client)
            time.sleep(1/240)

            if np.linalg.norm(get_drone_pose()[:YAW]-path[-1], ord=2) < 0.15:
                break
'''
