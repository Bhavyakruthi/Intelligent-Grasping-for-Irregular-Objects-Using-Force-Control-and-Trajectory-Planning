
import numpy as np
from panda_env import PandaEnv

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Warning: networkx or matplotlib not installed. Skipping visualization.")
    VISUALIZATION_AVAILABLE = False
    def visualize_grasp_points(*args, **kwargs):
        pass

def admm_grasp_points(center, radius, max_gripper_width=0.08, num_iters=50, rho=0.1):
    """Compute antipodal grasp points using ADMM."""
    # Initialize points on opposite sides, accounting for center offset
    p1 = center + np.array([radius, 0, 0])
    p2 = center - np.array([radius, 0, 0])
    z1, z2 = p1.copy(), p2.copy()
    u1, u2 = np.zeros(3), np.zeros(3)
    epsilon = 1e-6
    for k in range(num_iters):
        # Update p1, p2 with antipodal regularization
        A = np.eye(3) * (1 + rho + 0.1)  # Add term to encourage antipodality
        b1 = p2 + rho * (z1 - u1) + 0.1 * (center - (p2 - center))  # Push p1 toward -(p2 - c)
        b2 = p1 + rho * (z2 - u2) + 0.1 * (center - (p1 - center))
        p1_new = np.linalg.solve(A, b1)
        p2_new = np.linalg.solve(A, b2)

        # Update z1, z2 (project onto sphere)
        z1_temp = p1_new + u1
        z2_temp = p2_new + u2
        diff1 = z1_temp - center
        diff2 = z2_temp - center
        norm1 = np.linalg.norm(diff1)
        norm2 = np.linalg.norm(diff2)
        
        # Robust projection
        if norm1 < epsilon:
            diff1 = np.array([radius, 0, 0])
            norm1 = radius
        if norm2 < epsilon:
            diff2 = np.array([-radius, 0, 0])
            norm2 = radius
        z1 = center + radius * diff1 / norm1
        z2 = center + radius * diff2 / norm2

        # Update dual variables
        p1, p2 = p1_new, p2_new
        u1 += p1 - z1
        u2 += p2 - z2

        # Check antipodality
        n1 = (p1 - center) / np.linalg.norm(p1 - center)
        n2 = (p2 - center) / np.linalg.norm(p2 - center)
        dot_product = np.dot(n1, -n2)
        dist = np.linalg.norm(p1 - p2)
        if dist <= max_gripper_width and dot_product >= 0.5 and np.linalg.norm(p1 - z1) < 1e-4 and np.linalg.norm(p2 - z2) < 1e-4:
            break

    # Final projection
    p1 = center + radius * (p1 - center) / max(np.linalg.norm(p1 - center), epsilon)
    p2 = center + radius * (p2 - center) / max(np.linalg.norm(p2 - center), epsilon)
    
    print(f"ADMM: p1={p1}, p2={p2}, dist={np.linalg.norm(p1 - p2)}")
    return p1, p2

def check_force_closure(p1, p2, center, mu=0.5):
    diff1 = p1 - center
    diff2 = p2 - center
    norm1 = np.linalg.norm(diff1)
    norm2 = np.linalg.norm(diff2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        print("Warning: Grasp point coincides with center, force closure invalid.")
        return False
    
    n1 = diff1 / norm1
    n2 = diff2 / norm2
    dot_product = np.dot(n1, -n2)
    print(f"Force closure: dot_product={dot_product}")
    return dot_product >= 0.8

def visualize_grasp_points(p1, p2, center, point_cloud, radius):
    if not VISUALIZATION_AVAILABLE:
        return
    G = nx.Graph()
    G.add_node("p1", pos=p1)
    G.add_node("p2", pos=p2)
    G.add_node("center", pos=center)
    G.add_edge("p1", "p2")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', s=1, label='Object')
    
    pos = nx.get_node_attributes(G, 'pos')
    for node, coord in pos.items():
        if node == "center":
            ax.scatter(coord[0], coord[1], coord[2], c='g', s=100, label='Centroid')
        else:
            ax.scatter(coord[0], coord[1], coord[2], c='r', s=100, label='Grasp Point' if node == "p1" else "")
    
    for edge in G.edges():
        node1, node2 = edge
        x = [pos[node1][0], pos[node2][0]]
        y = [pos[node1][1], pos[node2][1]]
        z = [pos[node1][2], pos[node2][2]]
        ax.plot(x, y, z, 'k-', label='Gripper' if edge[0] == "p1" else "")
    
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='c', alpha=0.2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("Grasp Points Visualization")
    plt.savefig(f"grasp_object_{len(plt.get_fignums()) + 1}.png")
    plt.close()

def process_result(process, panda_position, fingers, error=0.01):
    if process == 0:
        print("--Grasped a sphere--")
        return False, process + 1, fingers
    elif process == 1:
        print("--Moving to the box--")
        return False, process + 1, fingers
    elif process == 2:
        if abs(panda_position[2] - 0.15) < error and fingers == 0:
            print("--At release height, releasing--")
            return False, process, 1
        elif fingers == 1 and panda_position[2] > 0.4:
            print("--Release confirmed, moving to return--")
            return False, process + 1, fingers
        else:
            return False, process, fingers
    elif process == 3:
        print("--Returning to safe position--")
        if panda_position[2] > 0.4 and abs(panda_position[0] - 0.3) < error:
            print("--Safe position reached--")
            return False, process + 1, fingers
        return False, process, fingers
    elif process == 4:
        print("--Ready for next object--")
        return True, 0, fingers
    else:
        return False, process, fingers

def main():
    env = PandaEnv()
    error = 0.01

    k_p = 5
    k_d = 1.5
    dt = 1. / 240.

    panda_position = env.reset()

    for i in range(1, 4):
        isEnd = False
        process = 0
        fingers = 1
        try:
            object_center, object_radius, object_point_cloud = env.get_object_info(i-1)
            print(f"Object {i} center: {object_center}")
        except ValueError as e:
            print(f"Error: {e}")
            continue
        
        p1, p2 = admm_grasp_points(object_center, object_radius)
        
        if not check_force_closure(p1, p2, object_center):
            print(f"Warning: Grasp for object {i} may not be stable (force closure failed)")
            continue
        
        visualize_grasp_points(p1, p2, object_center, object_point_cloud, object_radius)
        
        object_position = (p1 + p2) / 2

        for t in range(300):
            env.render()

            target_x = object_position[0]
            target_y = object_position[1]
            target_z = object_position[2] - 0.006

            if fingers == 0:
                if (panda_position[3] + panda_position[4]) < error + 0.037:
                    if process == 0:
                        target_z = 0.5
                    elif process == 1:
                        target_x = 1.0
                        target_z = 0.5
                    elif process == 2:
                        target_z = 0.15
            else:
                if process == 2:
                    target_z = 0.5
                elif process == 3:
                    target_x = 0.3
                    target_z = 0.5
                elif process == 4:
                    target_z = 0.5

            dx = target_x - panda_position[0]
            dy = target_y - panda_position[1]
            dz = target_z - panda_position[2]

            if abs(dx) < error and abs(dy) < error and abs(dz) < error:
                if process == 0:
                    fingers = 0
                elif process == 3 or process == 4:
                    isEnd, process, fingers = process_result(process, panda_position, fingers, error)
                    if isEnd:
                        break

            pd_x = k_p * dx + k_d * dx / dt
            pd_y = k_p * dy + k_d * dy / dt
            pd_z = k_p * dz + k_d * dz / dt

            max_vel_xy = 50
            max_vel_z_down = 20.0
            max_vel_z_up = 50

            pd_x = max(min(pd_x, max_vel_xy), -max_vel_xy)
            pd_y = max(min(pd_y, max_vel_xy), -max_vel_xy)
            pd_z = max(min(pd_z, max_vel_z_up if dz >= 0 else max_vel_z_down), -max_vel_z_up if dz >= 0 else -max_vel_z_down)

            action = [pd_x, pd_y, pd_z, fingers]
            panda_position, reward, done, info = env.step(action, process, i-1)
            object_position = info['object_position']

            if done:
                isEnd, process, fingers = process_result(process, panda_position, fingers, error)
                if isEnd:
                    break

    env.close()

if __name__ == "__main__":
    main()