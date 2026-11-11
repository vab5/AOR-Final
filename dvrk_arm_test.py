#!/usr/bin/env python

# Author: Anton Deguet
# Date: 2015-02-22

# (C) Copyright 2015-2023 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

# Start a single arm using
# > rosrun dvrk_robot dvrk_console_json -j <console-file>

# To communicate with the arm using ROS topics, see the python based example dvrk_arm_test.py:
# > rosrun dvrk_python dvrk_arm_test.py -a <arm-name>

import argparse
import sys
import time
import crtk
import dvrk
import math
import numpy
import PyKDL
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os

# example of application using arm.py
class example_application:

    # configuration
    def __init__(self, ral, arm_name, expected_interval):
        print('configuring dvrk_arm_test for {}'.format(arm_name))
        self.ral = ral
        self.expected_interval = expected_interval
        self.arm = dvrk.arm(ral = ral,
                            arm_name = arm_name,
                            expected_interval = expected_interval)

    # homing example
    def home(self):
        self.arm.check_connections()

        print('starting enable')
        if not self.arm.enable(10):
            sys.exit('failed to enable within 10 seconds')
        print('starting home')
        if not self.arm.home(10):
            sys.exit('failed to home within 10 seconds')
        # get current joints just to set size
        print('move to starting position')
        goal = numpy.copy(self.arm.setpoint_jp())
        # go to zero position, for PSM and ECM make sure 3rd joint is past cannula
        goal.fill(0)
        if ((self.arm.name() == 'PSM1') or (self.arm.name() == 'PSM2')
            or (self.arm.name() == 'PSM3') or (self.arm.name() == 'ECM')):
            goal[2] = 0.12
        # move and wait
        print('moving to starting position')
        self.arm.move_jp(goal).wait()
        # try to move again to make sure waiting is working fine, i.e. not blocking
        print('testing move to current position')
        move_handle = self.arm.move_jp(goal)
        print('move handle should return immediately')
        move_handle.wait()
        print('home complete')

    # get methods
    def run_get(self):
        [p, v, e, t] = self.arm.measured_js()
        d = self.arm.measured_jp()
        [d, t] = self.arm.measured_jp(extra = True)
        d = self.arm.measured_jv()
        [d, t] = self.arm.measured_jv(extra = True)
        d = self.arm.measured_jf()
        [d, t] = self.arm.measured_jf(extra = True)
        d = self.arm.measured_cp()
        print('measured cp: ', d)
        [d, t] = self.arm.measured_cp(extra = True)
        d = self.arm.local.measured_cp()
        [d, t] = self.arm.local.measured_cp(extra = True)
        d = self.arm.measured_cv()
        [d, t] = self.arm.measured_cv(extra = True)
        d = self.arm.body.measured_cf()
        [d, t] = self.arm.body.measured_cf(extra = True)
        d = self.arm.spatial.measured_cf()
        [d, t] = self.arm.spatial.measured_cf(extra = True)

        [p, v, e, t] = self.arm.setpoint_js()
        d = self.arm.setpoint_jp()
        [d, t] = self.arm.setpoint_jp(extra = True)
        d = self.arm.setpoint_jv()
        [d, t] = self.arm.setpoint_jv(extra = True)
        d = self.arm.setpoint_jf()
        [d, t] = self.arm.setpoint_jf(extra = True)
        d = self.arm.setpoint_cp()
        [d, t] = self.arm.setpoint_cp(extra = True)
        d = self.arm.local.setpoint_cp()
        [d, t] = self.arm.local.setpoint_cp(extra = True)

    # direct joint control example
    def run_servo_jp(self):
        print('starting servo_jp')
        # get current position
        initial_joint_position = numpy.copy(self.arm.setpoint_jp())
        print('testing direct joint position for 2 joints out of %i' % initial_joint_position.size)
        amplitude = math.radians(5.0) # +/- 5 degrees
        duration = 5  # seconds
        samples = duration / self.expected_interval
        # create a new goal starting with current position
        goal_p = numpy.copy(initial_joint_position)
        goal_v = numpy.zeros(goal_p.size)
        start = time.time()

        sleep_rate = self.ral.create_rate(1.0 / self.expected_interval)
        for i in range(int(samples)):
            angle = i * math.radians(360.0) / samples
            goal_p[0] = initial_joint_position[0] + amplitude * (1.0 - math.cos(angle))
            goal_p[1] = initial_joint_position[1] + amplitude *  (1.0 - math.cos(angle))
            goal_v[0] = amplitude * math.sin(angle)
            goal_v[1] = goal_v[0]
            self.arm.servo_jp(goal_p, goal_v)
            sleep_rate.sleep()

        actual_duration = time.time() - start
        print('servo_jp complete in %2.2f seconds (expected %2.2f)' % (actual_duration, duration))

    # goal joint control example
    def run_move_jp(self):
        print('starting move_jp')
        # get current position
        initial_joint_position = numpy.copy(self.arm.setpoint_jp())
        print('testing goal joint position for 2 joints out of %i' % initial_joint_position.size)
        amplitude = math.radians(10.0)
        # create a new goal starting with current position
        goal = numpy.copy(initial_joint_position)
        # first motion
        goal[0] = initial_joint_position[0] + amplitude
        goal[1] = initial_joint_position[1] - amplitude
        self.arm.move_jp(goal).wait()
        # second motion
        goal[0] = initial_joint_position[0] - amplitude
        goal[1] = initial_joint_position[1] + amplitude
        self.arm.move_jp(goal).wait()

        d = self.arm.measured_cp()
        print('measured cp: ', d)

        # back to initial position
        # self.arm.move_jp(initial_joint_position).wait()
        print('move_jp complete')

    # utility to position tool/camera deep enough before cartesian examples
    def prepare_cartesian(self):
        # make sure the camera is past the cannula and tool vertical
        goal = numpy.copy(self.arm.setpoint_jp())
        if ((self.arm.name().endswith('PSM1')) or (self.arm.name().endswith('PSM2'))
            or (self.arm.name().endswith('PSM3')) or (self.arm.name().endswith('ECM'))):
            print('preparing for cartesian motion')
            # set in position joint mode
            goal[0] = 0.0
            goal[1] = 0.0
            goal[2] = 0.12
            goal[3] = 0.0
            self.arm.move_jp(goal).wait()

    # direct cartesian control example
    def run_servo_cp(self):
        print('starting servo_cp')
        self.prepare_cartesian()

        # create a new goal starting with current position
        initial_cartesian_position = PyKDL.Frame()
        initial_cartesian_position.p = self.arm.setpoint_cp().p
        initial_cartesian_position.M = self.arm.setpoint_cp().M
        goal = PyKDL.Frame()
        goal.p = self.arm.setpoint_cp().p
        goal.M = self.arm.setpoint_cp().M
        # motion parameters
        amplitude = 0.02 # 4 cm total
        duration = 5  # 5 seconds
        samples = duration / self.expected_interval
        start = time.time()

        sleep_rate = self.ral.create_rate(1.0 / self.expected_interval)
        for i in range(int(samples)):
            goal.p[0] =  initial_cartesian_position.p[0] + amplitude *  (1.0 - math.cos(i * math.radians(360.0) / samples))
            goal.p[1] =  initial_cartesian_position.p[1] + amplitude *  (1.0 - math.cos(i * math.radians(360.0) / samples))
            self.arm.servo_cp(goal)
            # check error on kinematics, compare to desired on arm.
            # to test tracking error we would compare to
            # current_position
            setpoint_cp = self.arm.setpoint_cp()
            errorX = goal.p[0] - setpoint_cp.p[0]
            errorY = goal.p[1] - setpoint_cp.p[1]
            errorZ = goal.p[2] - setpoint_cp.p[2]
            error = math.sqrt(errorX * errorX + errorY * errorY + errorZ * errorZ)
            if error > 0.002: # 2 mm
                print('Inverse kinematic error in position [%i]: %s (might be due to latency)' % (i, error))
            sleep_rate.sleep()

        actual_duration = time.time() - start
        print('servo_cp complete in %2.2f seconds (expected %2.2f)' % (actual_duration, duration))

    # direct cartesian control example
    def run_move_cp(self):
        print('starting move_cp')
        self.prepare_cartesian()

        # create a new goal starting with current position
        initial_cartesian_position = PyKDL.Frame()
        initial_cartesian_position.p = self.arm.setpoint_cp().p
        initial_cartesian_position.M = self.arm.setpoint_cp().M
        goal = PyKDL.Frame()
        goal.p = self.arm.setpoint_cp().p
        goal.M = self.arm.setpoint_cp().M

        # motion parameters
        amplitude = 0.05 # 5 cm

        d = self.arm.measured_cp()
        print('CP before 1st motion: ')
        print()
        print(d)
        print()

        print('Initial cartesian position p[0]: ')
        print()
        print(initial_cartesian_position.p[0])
        print()

        print('Initial cartesian position p[1]: ')
        print()
        print(initial_cartesian_position.p[1])
        print()

        # first motion
        goal.p[0] =  initial_cartesian_position.p[0] - amplitude
        goal.p[1] =  initial_cartesian_position.p[1]
        # self.arm.move_cp(goal).wait()

        print('Goal.P[0]: ')
        print()
        print(goal.p[0])
        print()

        print('Goal.P[1]: ')
        print()
        print(goal.p[1])
        print()

        d = self.arm.measured_cp()
        print('CP after 1st motion: ')
        print()
        print(d)
        print()

        # second motion
        goal.p[0] =  initial_cartesian_position.p[0] + amplitude
        goal.p[1] =  initial_cartesian_position.p[1]
        # self.arm.move_cp(goal).wait()
        # back to initial position
        goal.p[0] =  initial_cartesian_position.p[0]
        goal.p[1] =  initial_cartesian_position.p[1]
        # self.arm.move_cp(goal).wait()
        # first motion
        goal.p[0] =  initial_cartesian_position.p[0]
        goal.p[1] =  initial_cartesian_position.p[1] - amplitude
        # self.arm.move_cp(goal).wait()


        # second motion
        goal.p[0] =  initial_cartesian_position.p[0]
        goal.p[1] =  initial_cartesian_position.p[1] + amplitude
        self.arm.move_cp(goal).wait()
        # back to initial position
        goal.p[0] =  initial_cartesian_position.p[0]
        goal.p[1] =  initial_cartesian_position.p[1]
        self.arm.move_cp(goal).wait()

        d = self.arm.measured_cp()
        print()
        print('measured cp before: ', d)
        print()

        # new_rotation = PyKDL.Rotation(1, 0, 0,
        #                       0, 1, 0,
        #                       0, 0, 1)

        # goal.M = new_rotation

        print('Rotation matrix goal: ')
        print()
        print(goal.M)
        print()

        goal.p[0] = 0.1
        goal.p[1] = 0.1
        goal.p[2] = -0.1
        self.arm.move_cp(goal).wait()

        d = self.arm.measured_cp()
        print('measured cp after: ', d)
        print()

        print('move_cp complete')


    def drawV(self):
        class Node:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
                self.parent = None
                self.cost = 0

        class RRTStar:
            def __init__(self, start, goal, map_size, obstacles=None, step_size=0.001, max_iter=10000):
                self.start = Node(*start)
                self.goal = Node(*goal)
                self.map_size = map_size
                self.step_size = step_size
                self.max_iter = max_iter
                self.node_list = [self.start]
                self.goal_region_radius = 0.003
                self.search_radius = 0.005
                self.goal_bias = 0.4
                self.obstacles = obstacles if obstacles else []
                self.goal_reached = False
                self.path = []

            def get_random_node(self):
                if random.random() > self.goal_bias:
                    return Node(
                        random.uniform(-self.map_size[0], self.map_size[0]),
                        random.uniform(-self.map_size[1], self.map_size[1]),
                        random.uniform(-self.map_size[2], 0)
                    )
                else:
                    return Node(self.goal.x, self.goal.y, self.goal.z)

            def steer(self, from_node, to_node):
                vector = np.array([to_node.x - from_node.x, to_node.y - from_node.y, to_node.z - from_node.z])
                norm_vector = vector / np.linalg.norm(vector) * self.step_size
                new_node = Node(from_node.x + norm_vector[0], from_node.y + norm_vector[1], from_node.z + norm_vector[2])
                new_node.cost = from_node.cost + self.step_size
                new_node.parent = from_node
                return new_node

            def is_collision_free(self, node):
                for (ox, oy, oz, size) in self.obstacles:
                    if (node.x - ox) ** 2 + (node.y - oy) ** 2 + (node.z - oz) ** 2 <= size ** 2:
                        return False
                return True

            def find_neighbors(self, new_node):
                return [node for node in self.node_list
                        if np.linalg.norm([node.x - new_node.x, node.y - new_node.y, node.z - new_node.z]) < self.search_radius]

            def choose_parent(self, neighbors, nearest_node, new_node):
                min_cost = nearest_node.cost + np.linalg.norm([new_node.x - nearest_node.x, new_node.y - nearest_node.y, new_node.z - nearest_node.z])
                best_node = nearest_node
                for neighbor in neighbors:
                    cost = neighbor.cost + np.linalg.norm([new_node.x - neighbor.x, new_node.y - neighbor.y, new_node.z - neighbor.z])
                    if cost < min_cost and self.is_collision_free(neighbor):
                        best_node = neighbor
                        min_cost = cost
                new_node.cost = min_cost
                new_node.parent = best_node
                return new_node

            def rewire(self, new_node, neighbors):
                for neighbor in neighbors:
                    cost = new_node.cost + np.linalg.norm([neighbor.x - new_node.x, neighbor.y - new_node.y, neighbor.z - new_node.z])
                    if cost < neighbor.cost and self.is_collision_free(neighbor):
                        neighbor.parent = new_node
                        neighbor.cost = cost

            def reached_goal(self, node):
                return np.linalg.norm([node.x - self.goal.x, node.y - self.goal.y, node.z - self.goal.z]) < self.goal_region_radius

            def generate_final_path(self, goal_node):
                path = []
                node = goal_node
                while node:
                    path.append([float(node.x), float(node.y), float(node.z)])
                    node = node.parent
                return path[::-1]

            def plan(self):
                for _ in range(self.max_iter):
                    rand_node = self.get_random_node()
                    nearest_node = self.get_nearest_node(rand_node)
                    new_node = self.steer(nearest_node, rand_node)
                    if self.is_collision_free(new_node):
                        neighbors = self.find_neighbors(new_node)
                        new_node = self.choose_parent(neighbors, nearest_node, new_node)
                        self.node_list.append(new_node)
                        self.rewire(new_node, neighbors)
                        if self.reached_goal(new_node):
                            self.goal_reached = True
                            self.path = self.generate_final_path(new_node)
                            return self.path
                return None

            def get_nearest_node(self, rand_node):
                distances = [np.linalg.norm([node.x - rand_node.x, node.y - rand_node.y, node.z - rand_node.z]) for node in self.node_list]
                return self.node_list[np.argmin(distances)]

        def draw_final_path(ax, path, waypoints, obstacles):
            for i, point in enumerate(waypoints):
                if i == 0:
                    ax.plot([point[0]], [point[1]], [point[2]], 'bo', label='Start')
                elif i == len(waypoints) - 1:
                    ax.plot([point[0]], [point[1]], [point[2]], 'go', label='Goal')
                else:
                    ax.plot([point[0]], [point[1]], [point[2]], 'ro')

            for (ox, oy, oz, size) in obstacles:
                u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
                x = ox + size * np.cos(u) * np.sin(v)
                y = oy + size * np.sin(u) * np.sin(v)
                z = oz + size * np.cos(v)
                ax.plot_surface(x, y, z, color='r', alpha=0.3)

            xs, ys, zs = zip(*path)
            ax.plot(xs, ys, zs, color='g', linewidth=2, label='Full Path')
            ax.legend()
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.grid(True)


        def rrt():
            os.makedirs("results", exist_ok=True)

            waypoints = [
                [0, 0, -0.1],
                [0.05, -0.05, -0.11],
                [0.1, 0.0, -0.12]
            ]

            map_size = [0.25, 0.25, 0.25]
            all_path = []
            obstacles = []

            # Optional: add obstacles at midpoints
            for i in range(1, len(waypoints)):
                mid = [(waypoints[i-1][j] + waypoints[i][j]) / 2 for j in range(3)]
                obstacles.append((*mid, 0.01))

            for i in range(len(waypoints) - 1):
                start = waypoints[i]
                goal = waypoints[i + 1]
                planner = RRTStar(start, goal, map_size, obstacles=obstacles)
                path = planner.plan()
                if not path:
                    print(f"Failed to find path from {start} to {goal}")
                    exit()
                if i > 0:
                    path = path[1:]  # Remove duplicate connection
                all_path.extend(path)



            # Save path to file
            np.savetxt("results/final_path.txt", np.array(all_path), fmt="%.5f", header="x y z")

            # Plot and save image
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            draw_final_path(ax, all_path, waypoints, obstacles)
            plt.savefig("results/final_path_plot.png", dpi=300)
            print("Final path saved to results/final_path.txt and plotted to results/final_path_plot.png")


            return all_path

        self.prepare_cartesian()

        # # create a new goal starting with current position
        # initial_cartesian_position = PyKDL.Frame()
        # initial_cartesian_position.p = self.arm.setpoint_cp().p
        # initial_cartesian_position.M = self.arm.setpoint_cp().M
        # goal = PyKDL.Frame()
        # goal.p = self.arm.setpoint_cp().p
        # goal.M = self.arm.setpoint_cp().M

        # print('Original Position: ')
        # print()
        # print(goal.p[0])
        # print(goal.p[1])
        # print(goal.p[2])
        # print()

        # #second point in the V:
        # goal.p[0] =  initial_cartesian_position.p[0] + 0.05
        # goal.p[1] =  initial_cartesian_position.p[1] - 0.05
        # self.arm.move_cp(goal).wait()

        # #third point in the V:
        # goal.p[0] =  goal.p[0] + 0.05
        # goal.p[1] =  goal.p[1] + 0.05
        # self.arm.move_cp(goal).wait()

        points = rrt()

        for point in points:
            goal = PyKDL.Frame()
            goal.p = PyKDL.Vector(point[0], point[1], point[2])
            goal.M = self.arm.setpoint_cp().M  # Keep the orientation the same
            self.arm.move_cp(goal).wait()

            # Print the coordinates after moving to each point
            print('Moved to point:')
            print(f"x: {goal.p[0]}, y: {goal.p[1]}, z: {goal.p[2]}")


    # main method
    def run(self):
        self.home()
        # self.run_get()
        # self.run_servo_jp()
        # self.run_move_jp()
        # self.run_servo_cp()
        # self.run_move_cp()
        self.drawV()

if __name__ == '__main__':
    # extract ros arguments (e.g. __ns:= for namespace)
    argv = crtk.ral.parse_argv(sys.argv[1:]) # skip argv[0], script name

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arm', type=str, required=True,
                        choices=['ECM', 'MTML', 'MTMR', 'PSM1', 'PSM2', 'PSM3'],
                        help = 'arm name corresponding to ROS topics without namespace.  Use __ns:= to specify the namespace')
    parser.add_argument('-i', '--interval', type=float, default=0.01,
                        help = 'expected interval in seconds between messages sent by the device')
    args = parser.parse_args(argv)

    ral = crtk.ral('dvrk_arm_test')
    application = example_application(ral, args.arm, args.interval)
    ral.spin_and_execute(application.run)
