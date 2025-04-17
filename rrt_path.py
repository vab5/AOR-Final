import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random
import os

# Node class representing a state in the space
class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.cost = 0

# RRT* algorithm
class RRTStar:
    def __init__(self, start, goal, num_random_obstacles, map_size, step_size=0.001, max_iter=1000):
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
        self.map_size = map_size
        self.obstacles = self.generate_random_obstacles(num_random_obstacles)
        self.step_size = step_size
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.goal_region_radius = 0.003
        self.search_radius = 0.005
        self.path = None
        self.goal_reached = False
        self.goal_bias = 0.4

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setup_visualization()

    def generate_random_obstacles(self, num_obstacles):
        obstacles = []
        for _ in range(num_obstacles):
            ox = random.uniform(0.02, self.map_size[0] - 0.02)
            oy = random.uniform(0.02, self.map_size[1] - 0.02)
            oz = random.uniform(-self.map_size[2] + 0.02, -0.02)
            size = random.uniform(0.01, 0.03)
            obstacles.append((ox, oy, oz, size))
        return obstacles

    def generate_obstacle(self, x, y, z, size=0.01):
        self.obstacles.append((x, y, z, size))

    def setup_visualization(self):
        for i, point in enumerate(waypoints):
            if i == 0:
                self.ax.plot([point[0]], [point[1]], [point[2]], 'bo', label='Start')
                self.ax.text(point[0], point[1], point[2], f'Start', fontsize=8, color='blue')
            elif i == len(waypoints) - 1:
                self.ax.plot([point[0]], [point[1]], [point[2]], 'go', label='Goal')
                self.ax.text(point[0], point[1], point[2], f'Goal', fontsize=8, color='green')
            else:
                self.ax.plot([point[0]], [point[1]], [point[2]], 'ro')
                self.ax.text(point[0], point[1], point[2], f'P{i}', fontsize=8, color='red')

        self.ax.set_xlim(-self.map_size[0], self.map_size[0])
        self.ax.set_ylim(-self.map_size[1], self.map_size[1])
        self.ax.set_zlim(-self.map_size[2], 0)
        self.ax.grid(True)
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        self.draw_obstacles()

    def draw_obstacles(self):
        for (ox, oy, oz, size) in self.obstacles:
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = ox + size * np.cos(u) * np.sin(v)
            y = oy + size * np.sin(u) * np.sin(v)
            z = oz + size * np.cos(v)
            self.ax.plot_surface(x, y, z, color='r', alpha=0.3)

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
        while node is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        return path[::-1]

    def get_nearest_node(self, node_list, rand_node):
        distances = [np.linalg.norm([node.x - rand_node.x, node.y - rand_node.y, node.z - rand_node.z]) for node in node_list]
        return node_list[np.argmin(distances)]

    def draw_tree(self, node):
        if node.parent:
            self.ax.plot([node.x, node.parent.x], [node.y, node.parent.y], [node.z, node.parent.z], "-b")

    def draw_path(self):
        if self.path:
            self.ax.plot([x[0] for x in self.path], [x[1] for x in self.path], [x[2] for x in self.path], '-g', label='Path')




if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    waypoints = [
        [0, 0, -0.1],
        [0.05, -0.05, -0.11],
        [0.1, 0.0, -0.12]
    ]
    waypoint_index = 0
    segment_path = []
    full_path = []
    planning_complete = False

    rrt_star = RRTStar(waypoints[0], waypoints[1], num_random_obstacles=0, map_size=[0.25, 0.25, 0.25])

    def animate(i):
        global waypoint_index, segment_path, full_path, planning_complete

        if planning_complete:
            return

        if not rrt_star.goal_reached:
            rand_node = rrt_star.get_random_node()
            nearest_node = rrt_star.get_nearest_node(rrt_star.node_list, rand_node)
            new_node = rrt_star.steer(nearest_node, rand_node)

            if rrt_star.is_collision_free(new_node):
                neighbors = rrt_star.find_neighbors(new_node)
                new_node = rrt_star.choose_parent(neighbors, nearest_node, new_node)
                rrt_star.node_list.append(new_node)
                rrt_star.rewire(new_node, neighbors)
                rrt_star.draw_tree(new_node)

            if rrt_star.reached_goal(new_node):
                segment_path = rrt_star.generate_final_path(new_node)
                if waypoint_index > 0:
                    segment_path = segment_path[1:]
                full_path.extend(segment_path)
                rrt_star.draw_path()
                rrt_star.goal_reached = True

        else:
            waypoint_index += 1
            if waypoint_index >= len(waypoints) - 1:
                if full_path:
                    xs, ys, zs = zip(*full_path)
                    rrt_star.ax.plot(xs, ys, zs, color='g', linewidth=2, label='Full Path')
                    plt.legend()
                    np.savetxt("results/final_path.txt", np.array(full_path), fmt="%.5f", header="x y z")
                planning_complete = True
            else:
                rrt_star.start = Node(*segment_path[-1])
                rrt_star.goal = Node(*waypoints[waypoint_index + 1])
                rrt_star.node_list = [rrt_star.start]
                rrt_star.goal_reached = False

    def generate_obstacles_at_midpoints(rrt): 

        for i in range(1, len(waypoints)): 

            mid_x = (waypoints[i - 1][0] + waypoints[i][0]) / 2
            mid_y = (waypoints[i - 1][1] + waypoints[i][1]) / 2
            mid_z = (waypoints[i - 1][2] + waypoints[i][2]) / 2
            rrt_star.generate_obstacle(mid_x, mid_y, mid_z, 0.015)

        rrt_star.draw_obstacles()

    generate_obstacles_at_midpoints(rrt_star)

    ani = animation.FuncAnimation(rrt_star.fig, animate, frames=rrt_star.max_iter * len(waypoints), interval=10, repeat=False)
    ani.save("results/rrt_star_multi_segment.gif", writer="pillow", fps=30)
    plt.show()
