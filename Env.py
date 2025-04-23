
import pybullet as p
import pybullet_data
import math
import numpy as np

class PandaEnv:
    def __init__(self):
        self.step_counter = 0
        try:
            p.connect(p.GUI)
        except:
            print("GUI connection failed, using DIRECT.")
            p.connect(p.DIRECT)
        p.resetDebugVisualizerCamera(cameraDistance=0.7, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])
        self.object_centers = []
        self.object_radii = []
        self.object_point_clouds = []

    def duck_position(self, i):
        self.duck = i - 1
        duckpos, _ = p.getBasePositionAndOrientation(self.objectUid[self.duck])
        return duckpos

    def get_object_info(self, i):
        if i < 0 or i >= len(self.objectUid):
            raise ValueError(f"Invalid object index: {i}")
        self.duck = i
        center, _ = p.getBasePositionAndOrientation(self.objectUid[self.duck])
        radius = 0.027
        points = []
        num_samples = 100
        for _ in range(num_samples):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            x = center[0] + radius * np.sin(phi) * np.cos(theta)
            y = center[1] + radius * np.sin(phi) * np.sin(theta)
            z = center[2] + radius * np.cos(phi)
            points.append([x, y, z])
        return np.array(center), radius, np.array(points)

    def step(self, action, process, duck):
        self.duck = duck
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi/2.])

        dv = 0.003
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        currentPose = p.getLinkState(self.pandaUid, 11)
        self.currentPosition = currentPose[0]
        newPosition = [self.currentPosition[0] + dx, self.currentPosition[1] + dy, self.currentPosition[2] + dz]

        if process == 2 and fingers == 1:
            newPosition[2] = max(newPosition[2], 0.2)

        jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, newPosition, orientation)[0:7]
        p.setJointMotorControlArray(self.pandaUid, list(range(7)) + [9, 10], p.POSITION_CONTROL, list(jointPoses) + 2 * [fingers])
        p.stepSimulation()

        if self.check_for_collision():
            print("! Collision detected - adjusting position!")
            newPosition[2] = max(newPosition[2], 0.4)
            jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, newPosition, orientation)[0:7]
            p.setJointMotorControlArray(self.pandaUid, list(range(7)), p.POSITION_CONTROL, jointPoses)
            p.stepSimulation()

        state_goal_object, _ = p.getBasePositionAndOrientation(self.objectUid[self.duck])
        local_goal_object = self.objectUid[self.duck]
        if process == 1 or process == 2:
            local_goal_object = self.trayUid

        state_object, _ = p.getBasePositionAndOrientation(local_goal_object)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0])

        done = self.processDoneCondition(process, state_goal_object, state_robot)

        self.step_counter += 1
        info = {'object_position': state_object}
        self.panda_position = state_robot + state_fingers
        reward = 0
        return np.array(self.panda_position).astype(np.float32), reward, done, info

    def processDoneCondition(self, process, state_goal_object, state_robot):
        done = False
        if process == 0:
            if state_goal_object[2] > 0.4:
                done = True
        elif process == 1:
            if state_goal_object[0] > 0.65:
                done = True
        elif process == 2:
            if state_goal_object[0] > 0.6 and state_robot[2] <= 0.15:
                done = True
        elif process == 3:
            if state_robot[2] > 0.4 and state_robot[0] < 0.4:
                done = True
        elif process == 4:
            done = True
        return done

    def reset(self):
        duckcount = 3
        dis = 0.45 / duckcount

        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.pandaUid = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.tableUid = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])
        self.plainID = p.loadURDF("plane.urdf", basePosition=[0, 0, -0.65])
        self.trayUid = p.loadURDF("tray/traybox.urdf", basePosition=[1., 0, 0])

        self.objectUid = []
        self.object_centers = []
        self.object_radii = []
        self.object_point_clouds = []
        for i in range(duckcount):
            sphere_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.027)
            sphere_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.027, rgbaColor=[1, 0, 0, 1])
            sphere_pos = [0.55 - dis * i, 0, 0]
            sphere = p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=sphere_collision_shape, 
                                       baseVisualShapeIndex=sphere_visual_shape, basePosition=sphere_pos)
            self.objectUid.append(sphere)
            p.changeDynamics(sphere, -1, lateralFriction=2.0, spinningFriction=2.0, rollingFriction=0.001)
            p.resetBasePositionAndOrientation(sphere, sphere_pos, [0, 0, 0, 1])  # Ensure exact position
            center = sphere_pos
            radius = 0.027
            self.object_centers.append(center)
            self.object_radii.append(radius)
            points = []
            num_samples = 100
            for _ in range(num_samples):
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                x = center[0] + radius * np.sin(phi) * np.cos(theta)
                y = center[1] + radius * np.sin(phi) * np.sin(theta)
                z = center[2] + radius * np.cos(phi)
                points.append([x, y, z])
            self.object_point_clouds.append(np.array(points))

        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        for i in range(7):
            p.resetJointState(self.pandaUid, i, rest_poses[i])
        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid, 10, 0.08)

        self.state_robot = p.getLinkState(self.pandaUid, 11)[0]
        self.orien_robot = p.getLinkState(self.pandaUid, 11)[1]
        self.currentPosition = self.state_robot
        state_fingers = (p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0])
        self.panda_position = self.state_robot + state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        return np.array(self.panda_position).astype(np.float32)
    
    def render(self):
        dis = 1000.0
        yaw = p.getEulerFromQuaternion(self.orien_robot)[-1]

        xCamera = self.currentPosition[0]
        yCamera = self.currentPosition[1]
        zCamera = self.currentPosition[2] + 0.15

        xTarget = xCamera + math.cos(yaw) * dis
        yTarget = yCamera + math.sin(yaw) * dis
        zTarget = zCamera + math.tan(yaw) * dis

        view_matrix = p.computeViewMatrix(cameraEyePosition=[xCamera, yCamera, zCamera],
                                          cameraTargetPosition=[xTarget, yTarget, zTarget],
                                          cameraUpVector=[0., 0., 0.5])
        proj_matrix = p.computeProjectionMatrixFOV(fov=80,
                                                   aspect=float(960) / 720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.panda_position

    def close(self):
        p.disconnect()

    def check_for_collision(self):
        tray_collision = p.getContactPoints(self.pandaUid, self.trayUid)
        table_collision = p.getContactPoints(self.pandaUid, self.tableUid)
        return bool(tray_collision or table_collision)