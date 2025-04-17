import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os



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
        obstacles.append((*mid, 0.004))

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

if __name__ == "__main__": 
    print(rrt())