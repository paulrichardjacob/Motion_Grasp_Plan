import pybullet as p
import time
import sys
import random
import pybullet_data
import numpy as np
import burg_toolkit as burg
import pybullet_planning as pp
from scipy.stats import qmc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pybullet_planning.interfaces.robots.collision import get_collision_fn
from pybullet_planning.motion_planners import smooth_path
from pybullet_planning.motion_planners.utils import waypoints_from_path


#utility functions

# defining the workspace for the robot
cube_size = 0.25
cube_center = np.array([0.6, 0, 0.9])  # Position of the center of the cube
half_size = cube_size * 4
corner_points = [
    cube_center + np.array([x, y, z])
    for x in [-half_size, half_size]
    for y in [-half_size, half_size]
    for z in [-half_size, half_size]
]
def closest_point_in_segment(p, a, b):
    """
    Compute the closest point on a line segment to a given point.

    Parameters:
    - p (numpy.array): The point for which the closest point on the segment is to be found.
    - a (numpy.array): The starting point of the line segment.
    - b (numpy.array): The ending point of the line segment.

    Returns:
    - numpy.array: The closest point on the segment to the given point 'p'.

    Note:
    The function assumes that 'p', 'a', and 'b' are numpy arrays of the same dimension.
    """
    ab = b - a  # Vector from a to b
    t = np.dot(p - a, ab) / np.dot(ab, ab)  # Projection factor
    t = np.clip(t, 0, 1)  # Clipping to ensure the point lies on the segment
    closest_point = a + t * ab  # Compute the closest point
    return closest_point

def segment_intersects_cube(a, b, cube_min, cube_max):
    """
    Determine if a line segment intersects with a 3D cube.

    Parameters:
    - a (numpy.array): The starting point of the line segment.
    - b (numpy.array): The ending point of the line segment.
    - cube_min (numpy.array): The minimum corner (vertex) of the cube.
    - cube_max (numpy.array): The maximum corner (vertex) of the cube.

    Returns:
    - bool: True if the segment intersects the cube, False otherwise.

    Note:
    The function assumes that 'a', 'b', 'cube_min', and 'cube_max' are numpy arrays of dimension 3.
    """

    # Check if both endpoints of the segment are outside the cube on any dimension
    for i in range(3):
        if (a[i] < cube_min[i] and b[i] < cube_min[i]) or (a[i] > cube_max[i] and b[i] > cube_max[i]):
            return False

    # Check if any point of the segment lies inside the cube
    for i in range(3):
        if cube_min[i] <= a[i] <= cube_max[i] or cube_min[i] <= b[i] <= cube_max[i]:
            continue
        # Find the closest point on the segment to the cube's face
        closest_on_segment = closest_point_in_segment(np.array([cube_min[i], 0, 0]), a, b)[i]
        if cube_min[i] <= closest_on_segment <= cube_max[i]:
            return True

    return False

def is_inside_cube(position, cube_center, half_size):
    """
    Determine if a point is inside a 3D cube.

    Parameters:
    - position (tuple or list): The 3D coordinates of the point.
    - cube_center (tuple or list): The 3D coordinates of the cube's center.
    - half_size (float): Half the length of one side of the cube.

    Returns:
    - bool: True if the point is inside the cube, False otherwise.

    Note:
    The function assumes that 'position' and 'cube_center' are either tuples or lists of length 3.
    """

    # Check if the point's coordinates are within the bounds of the cube on all dimensions
    return all(cube_center[i] - half_size <= position[i] <= cube_center[i] + half_size for i in range(3))

def classify_point(point, obstacles, safety_distance):
    """
    Classify a point based on its proximity to obstacles.

    Parameters:
    - point (tuple or list): The 3D coordinates of the point to be classified.
    - obstacles (list): A list of 3D coordinates representing the obstacles.
    - safety_distance (float): The minimum distance from an obstacle to be considered safe.

    Returns:
    - str: A classification of the point as either "free point", "adjacent point", or "obstacle point".

    Note:
    The function assumes that 'point' and the elements of 'obstacles' are either tuples or lists of length 3.
    """

    # Initialize the minimum distance to a large value
    min_distance = float('inf')

    # Compute the distance from the point to each obstacle and update the minimum distance
    for obstacle in obstacles:
        distance = np.linalg.norm(np.array(point) - np.array(obstacle))
        min_distance = min(min_distance, distance)

    # Classify the point based on its proximity to the obstacles
    if min_distance > safety_distance:
        return "free point"
    elif min_distance == safety_distance:
        return "adjacent point"
    else:
        return "obstacle point"

def matrix_to_robot_config(robot, end_effector_link, matrix):
    # Convert 4x4 matrix to position and orientation using the burg toolkit
    position, orientation = burg.util.position_and_quaternion_from_tf(matrix, convention='pybullet')
    # Calculate the inverse kinematics to get joint configuration for the given pose
    joint_positions = p.calculateInverseKinematics(robot, end_effector_link, position, orientation, maxNumIterations=300, residualThreshold=0.00001)
    joint_positions = list(joint_positions)[:7]  # Assuming Franka Panda's 7 joints
    return joint_positions,position,orientation

def joint_to_cartesian(joint_angles, robot, end_effector_link):
    """
    Convert joint angles to the Cartesian position of the end effector.

    Parameters:
    - joint_angles: List of joint angles.
    - robot: The robot's body unique ID (from `p.loadURDF` or similar).
    - end_effector_link: The link index of the robot's end effector.

    Returns:
    - Cartesian position of the end effector as (x, y, z).
    """

    # Set the robot's joint angles
    for i, angle in enumerate(joint_angles):
        p.resetJointState(robot, i, angle)

    # Get the position of the end effector
    state = p.getLinkState(robot, end_effector_link)
    pos = state[0]  # The first item in the returned tuple is the Cartesian position

    return pos

def sample_point_along_z(orientation_matrix, world_grasp_position ,z_min_offset=0.09, z_max_offset=0.2):
    """
    Sample a point with a constant x, y and variable z along a line in the z-axis.

    Parameters:
    - orientation_matrix: The orientation matrix to determine the direction.
    - z_min_offset: The minimum offset added to the z-coordinate of the world grasp position.
    - z_max_offset: The maximum offset added to the z-coordinate of the world grasp position.

    Returns:
    - transformation_matrix: A 4x4 matrix representing the sampled point's position and orientation.
    """
    x = world_grasp_position[0, 3]
    y = world_grasp_position[1, 3]
    z_min = world_grasp_position[2, 3] + z_min_offset
    z_max = world_grasp_position[2, 3] + z_max_offset
    z = random.uniform(z_min, z_max)

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = orientation_matrix[:3, :3]
    transformation_matrix[:3, 3] = [x, y, z]

    return transformation_matrix

def get_euclidean_distance_fn(weights):
    """
    Generate a weighted Euclidean distance function between two configurations.

    Parameters:
    - weights (list or numpy array): Weights for each dimension of the configuration space.

    Returns:
    - function: A function that computes the weighted Euclidean distance between two configurations.
    """

    def distance_fn(q1, q2):
        diff = np.array(q2) - np.array(q1)
        return np.sqrt(np.dot(weights, diff * diff))

    return distance_fn

# Link indices for the Panda robot
def get_link_indices(robot, link_names):
    all_links = [p.getJointInfo(robot, i)[12].decode('UTF-8') for i in range(p.getNumJoints(robot))]
    return [all_links.index(name) for name in link_names]

# List of link pairs for which self-collision checks should be disabled.
# These pairs represent adjacent links in the Panda robot's kinematic chain.
# Disabling self-collision checks for these pairs can prevent false positives
# during motion planning, as these links are designed to be close to each other.
panda_self_collision_disabled_link_names = [
        ('panda_link1', 'panda_link2'),
        ('panda_link2', 'panda_link3'),
        ('panda_link3', 'panda_link4'),
        ('panda_link4', 'panda_link5'),
        ('panda_link5', 'panda_link6'),
        ('panda_link6', 'panda_link7'),
        ('panda_link7', 'panda_link8')
    ]

def sample_fn(obstacles, safety_distance):
    """
    Generate a sample configuration using Sobol sequence and classify its end effector position.

    Parameters:
    - obstacles (list): A list of 3D coordinates representing the obstacles.
    - safety_distance (float): The minimum distance from an obstacle to be considered safe.

    Returns:
    - list: A scaled sample of joint configurations.

    Note:
    The function uses a Sobol sequence to generate quasi-random samples in the configuration space.
    The samples are then scaled to the robot's joint ranges. The end effector position corresponding
    to each sample is checked to determine if it lies inside a specified cube. If it does, the position
    is classified based on its proximity to the obstacles. The function returns once a valid sample is found
    or after a timeout of 3 seconds.
    """

    # Create a list to store the samples
    samples_list = []

    # Initialize sobol sequence generator
    sobol_seq = qmc.Sobol(len(joint_indices), scramble=True)
    sobol_seq.random_state = np.random.RandomState(1)

    start_time = time.time()  # Record the start time
    while True:
        # Check if 10 seconds have passed
        if time.time() - start_time > 3:
            return sample_fn(obstacles, safety_distance)  # Re-run the sample_fn
        # Generate a single Sobol sequence sample
        sample = sobol_seq.random(n=1)[0]
        # Scale the sample to the joint ranges
        joint_ranges = [p.getJointInfo(robot, i)[8:10] for i in joint_indices]
        scaled_sample = [(high - low) * s + low for s, (low, high) in zip(sample, joint_ranges)]

        # Convert the joint sample to an end effector position
        end_effector_position, _ = p.getLinkState(robot, end_effector_link)[:2]
        if is_inside_cube(end_effector_position, cube_center, half_size):
            classification = classify_point(end_effector_position, obstacles, safety_distance)
            samples_list.append((scaled_sample, classification))
            return scaled_sample

def is_inside_box(position, min_corner, max_corner):
    return all(min_corner[i] <= position[i] <= max_corner[i] for i in range(3))

def sample_line(segment, step_size=.02, min_corner=None, max_corner=None, collision_fn=None):
    (q1, q2) = segment

    # Convert q1 and q2 to Cartesian space
    p1 = joint_to_cartesian(q1, robot, end_effector_link)
    p2 = joint_to_cartesian(q2, robot, end_effector_link)

    if min_corner is not None and max_corner is not None:
        # Check the Cartesian segment against the cube
        if not segment_intersects_cube(np.array(p1), np.array(p2), np.array(min_corner), np.array(max_corner)):
            return

    diff = np.array(q2) - np.array(q1)
    dist = np.linalg.norm(diff)
    for l in np.arange(0., dist, step_size):
        q = tuple(np.array(q1) + l * diff / dist)
        end_effector_position, _ = p.getLinkState(robot, end_effector_link)[:2]

        # Check if the end effector position is inside the specified volume (box)
        if (min_corner is not None) and (max_corner is not None) and not is_inside_box(end_effector_position, min_corner, max_corner):
            break

        # Check if the configuration is in collision
        if collision_fn and collision_fn(q):
            continue  # Skip this configuration if it's in collision

        yield q

    # Check the last point for collision before yielding
    if not (collision_fn and collision_fn(q2)):
        yield q2


def get_extend_fn(obstacles=[], min_corner=None, max_corner=None):
    collision_fn = get_collision_fn(robot, joint_indices, obstacles=obstacles,
                                    self_collisions=True,
                                    disabled_collisions=panda_self_collision_disabled_link_indices)

    roadmap = []

    def extend_fn(q1, q2):
        #print("Starting to extend from q1 to q2...")  # Print message at the start
        path = [q1]
        for q in sample_line(segment=(q1, q2), min_corner=min_corner, max_corner=max_corner):
            if collision_fn(q,safety_distance=0.1):
                return []
            path.append(q)
        #print("Completed extending the line.")  # Print message at the end
        return path

    return extend_fn, roadmap

def open_gripper():
    p.setJointMotorControl2(robot, 9, p.POSITION_CONTROL, targetPosition=0.04)  # Open the first finger
    p.setJointMotorControl2(robot, 10, p.POSITION_CONTROL, targetPosition=0.04)  # Open the second finger
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)


# Function to close the gripper
vel = -0.1
force = 800
def close_gripper():
    # Set the lead finger (joint 9) to velocity control mode
    p.setJointMotorControl2(robot, 9, p.VELOCITY_CONTROL, targetVelocity=vel, force=force)

    for _ in range(1000):  # You can adjust the number of steps as needed
        # Get the current position of the lead finger (joint 9)
        lead_finger_pos = p.getJointState(robot, 9)[0]

        # Set the follower finger (joint 10) to position control mode with the target position set to the lead finger's current position
        p.setJointMotorControl2(robot, 10, p.POSITION_CONTROL, targetPosition=lead_finger_pos, force=force,
                                targetVelocity=vel)

        # Step the simulation
        p.stepSimulation()
        time.sleep(1. / 240.)

def compute_grasp_for_world_frame(cube_position, grasp_pose_matrix):
    goal_pose = np.array([
        [1.0, 0.0, 0.0, cube_position[0]],
        [0.0, 1.0, 0.0, cube_position[1]],
        [0.0, 0.0, 1.0, cube_position[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])
    # second goal point
    second_goal_pose = np.array([
        [-1.0, 0.0, 0.0, position2[0]],
        [0.0, 1.0, 0.0, position2[1]],
        [0.0, 0.0, -1.0, position2[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Transformation from grasp to end-effector frame
    tf_grasp2ee = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    T_World = np.dot(goal_pose, grasp_pose_matrix)
    T_World2 = np.dot(T_World, second_goal_pose)
    world_grasp_position = np.dot(T_World, tf_grasp2ee)
    world_grasp_position2 = np.dot(T_World2, tf_grasp2ee)
    pos_ori_intermediate_point = sample_point_along_z(world_grasp_position, world_grasp_position)

    return pos_ori_intermediate_point, world_grasp_position ,world_grasp_position2


# Set up the environment for sampling
(p.connect(p.DIRECT))
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
plane = p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(2.5, 70, -25, [0, 0, 0.5])


# Set the robot and initial joint angles
robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
initial_joint_angles = [0.8214897102106269, -0.3627628274285913, 0.06249616451446385, -1.0671787864414042, 0.029315838964342797, 0.690071458587208, -1.4617677056424134]
for i, angle in enumerate(initial_joint_angles):
    p.resetJointState(bodyUniqueId=robot, jointIndex=i, targetValue=angle)
panda_self_collision_disabled_link_indices = [get_link_indices(robot, pair)
                                                  for pair in panda_self_collision_disabled_link_names]
# Load the object URDF
cube_path = 'C:/Users/paul_/OneDrive - Aston University/Dessertation Docs/Envirnment objects'
p.setAdditionalSearchPath(cube_path)
cube_position = [0.3, -0.4, 0.01]
cube = p.loadURDF("cube.urdf",cube_position )
# Load the cupboard URDF
cupboard_path = "C:/Users/paul_/OneDrive - Aston University/Dessertation Docs/Envirnment objects/cupboard/cupboard.urdf"
position = [0.3, -0.3, 0]
orientation = p.getQuaternionFromEuler([3.14159 / 2, 0, 3.14159 / 2])  # 90 degrees pitch and -90 degrees yaw rotation
cupboard_id = p.loadURDF(cupboard_path, basePosition=position, baseOrientation=orientation, globalScaling=0.001,useFixedBase=True)

# Get the joint indices of the robot
joint_indices = [i for i in range(p.getNumJoints(robot)) if p.getJointInfo(robot, i)[2] == p.JOINT_REVOLUTE]

# Create a list of obstacles
obstacles = [cupboard_id, plane]
# Define the collision checking function for the robot.
# This function will be used to determine if a given robot configuration is in collision with obstacles or itself.
# The self_collisions parameter is set to True, meaning the function will check for collisions between the robot's links.
collision_fn = get_collision_fn(robot, joint_indices, obstacles=obstacles,
                                self_collisions=True,
                                disabled_collisions=panda_self_collision_disabled_link_indices)
end_effector_link = 11  # Link index for the Panda's end effector
min_corner = corner_points[0]
max_corner = corner_points[-1]


# Gasp point for the cube , obtained from the burg toolkit
#But can be visualized and use that function to get the grasp point
"""
a = 2
def sample_grasp_poses():

        print("Inside sample_grasp_poses function")
        # Specify the 3D mesh file of the object you want to grasp
        mesh_fn = 'C:/Users/paul_/OneDrive - Aston University/Dessertation Docs/Envirnment objects/cube.stl'

        # Create an instance of the AntipodalGraspSampler class
        ags = burg.sampling.AntipodalGraspSampler()

        # Set the only_grasp_from_above attribute
        ags.only_grasp_from_above = True

        # Set the no_contact_below_z attribute
        ags.no_contact_below_z = 0.02  # Set your desired z threshold here

        ags.mesh = burg.io.load_mesh(mesh_fn)
        ags.gripper = burg.gripper.ParallelJawGripper(finger_length=0.05,opening_width=0.08, finger_thickness=0.003)

        # Sample a set of antipodal grasps
        graspset, contacts = ags.sample(100)
        print("Grasp sampling completed.")

        return ags, graspset  # returning both ags and graspset
        
ags, graspset = sample_grasp_poses()  # calling the function here to get ags and graspset
    # Visualize only the first grasp in the grasp set
for i, grasp in enumerate(graspset):
    print(f"Visualizing grasp {i+1}/{len(graspset)}")
    burg.visualization.show_grasp_set([ags.mesh], [grasp], gripper=ags.gripper, use_width=False,
                                      score_color_func=lambda s: [s, 1 - s, 0], with_plane=True)
    # Ask the user if they want to print the matrix for the current grasp
    print_choice = input("Do you want to print the 4x4 matrix for this grasp? (yes/no): ").strip().lower()

    if print_choice == 'yes':
        # Assuming the grasp object has a method or attribute to get its 4x4 matrix
        # If not, you might need to adjust this part based on the BURG toolkit's API
        grasp.pose
        grasp_pose_matrix = grasp.pose
        print(grasp_pose_matrix)

    # Move to the next grasp or exit the loop
    next_choice = input("Press Enter to view the next grasp or type 'exit' to end: ").strip().lower()
    if next_choice == 'exit':
        break

first_grasp = graspset[a]
first_grasp.pose
grasp_pose_matrix = first_grasp.pose"""

grasp_pose_matrix= np.array([
    [ 8.64539593e-02, -9.85964656e-01, -1.42826691e-01,  2.20326260e-02],
    [ 9.96181071e-01,  8.73112530e-02,  2.65946263e-04,  2.74999999e-02],
    [ 1.22081637e-02, -1.42304227e-01, 9.89747703e-01,  2.71826703e-02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
])

position2 = [0.3, -0.5, 0.2]


pos_ori_intermediate_point, world_grasp_position,world_grasp_position2 = compute_grasp_for_world_frame(cube_position, grasp_pose_matrix)
in_goal_joint_config, _, _ = matrix_to_robot_config(robot, end_effector_link, pos_ori_intermediate_point)
goal_joint_config,_,_= matrix_to_robot_config(robot, end_effector_link, world_grasp_position)
second_goal_joint_config,_,_ = matrix_to_robot_config(robot, end_effector_link, world_grasp_position2)

"""if collision_fn(goal_joint_config):
    print("The goal configuration is in collision!,Please! give anothor location")
    sys.exit("Exiting due to collision in goal configuration.")
else:
    print("The goal configuration is collision-free!")

if collision_fn(second_goal_joint_config):
    print("The second goal configuration is in collision!")
    sys.exit("Exiting due to collision in second goal configuration.")
else:
    print("The second goal configuration is collision-free!")"""

extend_fn, _ = get_extend_fn(obstacles, min_corner, max_corner)
weights = [1, 1, 1, 1, 1, 1, 1]
distance_fn = get_euclidean_distance_fn(weights=weights)
sample_fn_ags = lambda: sample_fn(obstacles, safety_distance=0.85)


inter_path = pp.prm(start=initial_joint_angles, goal=in_goal_joint_config, distance_fn=distance_fn,sample_fn=sample_fn_ags,
                                extend_fn=extend_fn, collision_fn=collision_fn,num_samples=300)

path = pp.prm(start=in_goal_joint_config, goal=goal_joint_config, distance_fn=distance_fn,sample_fn=sample_fn_ags,
                             extend_fn=extend_fn, collision_fn=collision_fn,num_samples=300)

second_path = pp.prm(start=goal_joint_config, goal=second_goal_joint_config, distance_fn=distance_fn, sample_fn=sample_fn_ags,
                    extend_fn=extend_fn, collision_fn=collision_fn, num_samples=300)

intr_smooth_path= smooth_path(inter_path, extend_fn, collision_fn, iterations=3)
smoothed_path = smooth_path(path, extend_fn, collision_fn, iterations=3)
second_smoothed_path = smooth_path(second_path, extend_fn, collision_fn, iterations=3)

intr_waypoints = waypoints_from_path(intr_smooth_path, tolerance=0.001)
waypoints = waypoints_from_path(smoothed_path, tolerance=0.001)
second_waypoints = waypoints_from_path(second_smoothed_path, tolerance=0.001)

p.disconnect()

# Visualize the path in 3D
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
plane = p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(2.5, 70, -25, [0, 0, 0.5])

# Set the robot and initial joint angles
robot = p.loadURDF("franka_panda/panda.urdf",[0,0,0],useFixedBase=True)
panda_self_collision_disabled_link_indices = [get_link_indices(robot, pair)
                                              for pair in panda_self_collision_disabled_link_names]
initial_joint_angles = [0.8214897102106269, -0.3627628274285913, 0.06249616451446385, -1.0671787864414042, 0.029315838964342797, 0.690071458587208, -1.4617677056424134]
for i, angle in enumerate(initial_joint_angles):
    p.resetJointState(bodyUniqueId=robot, jointIndex=i, targetValue=angle)
p.setAdditionalSearchPath(cube_path)
cube = p.loadURDF("cube.urdf",cube_position)
p.changeDynamics(cube, -1, lateralFriction=2.5, spinningFriction=1.5, rollingFriction=1.5)
initial_joint_angles = [0.8214897102106269, -0.3627628274285913, 0.06249616451446385, -1.0671787864414042, 0.029315838964342797, 0.690071458587208, -1.4617677056424134]
for i, angle in enumerate(initial_joint_angles):
    p.resetJointState(bodyUniqueId=robot, jointIndex=i, targetValue=angle)
cupboard = p.loadURDF(cupboard_path, basePosition=position, baseOrientation=orientation, globalScaling=0.001,useFixedBase=True)
MAX_VELOCITY = 0.08  # Adjust this value based on your needs

# Open the gripper

for conf in intr_waypoints:
    for j, value in enumerate(conf):
        p.setJointMotorControl2(robot, joint_indices[j], p.POSITION_CONTROL,
                                targetPosition=value, targetVelocity=MAX_VELOCITY)
        time.sleep(0.03)
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)
open_gripper()
# Move through waypoints
for conf in waypoints:
    for j, value in enumerate(conf):
        p.setJointMotorControl2(robot, joint_indices[j], p.POSITION_CONTROL,
                                targetPosition=value, targetVelocity=MAX_VELOCITY)
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)


close_gripper()

# Move through second_waypoints
for conf in second_waypoints:
    for j, value in enumerate(conf):
        p.setJointMotorControl2(robot, joint_indices[j], p.POSITION_CONTROL,
                                targetPosition=value, targetVelocity=MAX_VELOCITY)
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)