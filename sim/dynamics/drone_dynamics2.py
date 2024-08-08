
import numpy as np
from numba import jit, int32
from numba.experimental import jitclass
from numba.typed import List
from numba.types import ListType

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                        drone_dynamics2.py
                                                                        ------------------
                                        This is the secondary Python File, where all of the simulation is done. This is 
                                        done by solving 6 DoF Equations to get the state of the quadcopter after a certain
                                        input. (Testing purposes, right now using this instead of drone_dynamics.py)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""



"""
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                        NUMBA USAGE
                                                                        -----------
                                        Because I wanted to base all of my Code only on Python, I decided to use Numba
                                        as a way of accelerating the learning process. With the usage of Numba, similar
                                        speeds to C++ can be achieved, therefore unnecessary convertions with for example
                                        PYBIND11 are not needed. 

                                        Due to this, Code sometimes has to be structured or written a bit differently which
                                        sometimes results in some awkward line repetitions. (Some Numpy functionalities are
                                        not implemented.)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""




# Definiere die Spezifikationen für die Numba-Klasse
"""spec = [
    ('items', ListType(int32)),
    ('size', int32)
]

@jitclass(spec)
class Queue:
    def __init__(self):
        self.items = List.empty_list(int32)
        self.size = 0

    def is_empty(self):
        return self.size == 0

    def enqueue(self, item):
        self.items.append(item)
        self.size += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Dequeue from an empty queue")
        item = self.items[0]
        del self.items[0]
        self.size -= 1
        return item

    def get_size(self):
        return self.size

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from an empty queue")
        return self.items[0]
"""

@jit(nopython=True)
def quaternion_from_euler(roll, pitch, yaw):

    """
    Converts Euler angles (roll, pitch, yaw) to a quaternion.

    Args:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.

    Returns:
        np.ndarray: Quaternion [w, x, y, z] representing the rotation.
    """

    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])

@jit(nopython=True)
def quaternion_multiply(q1, q2):

    """
    Multiplies two quaternions.

    Args:
        q1 (np.ndarray): First quaternion [w, x, y, z].
        q2 (np.ndarray): Second quaternion [w, x, y, z].

    Returns:
        np.ndarray: Resultant quaternion from the multiplication [w, x, y, z].
    """

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

@jit(nopython=True)
def quaternion_rotate(q, v):

    """
    Rotates a vector by a quaternion.

    Args:
        q (np.ndarray): Quaternion [w, x, y, z] representing the rotation.
        v (np.ndarray): Vector to be rotated [vx, vy, vz].

    Returns:
        np.ndarray: Rotated vector [vx, vy, vz].
    """

    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    v_as_quat = np.array([0, v[0], v[1], v[2]])
    rotated_v = quaternion_multiply(quaternion_multiply(q, v_as_quat), q_conj)
    return rotated_v[1:]

@jit(nopython=True)
def quaternion_from_rotvec(rotvec):

    """
    Converts a rotation vector to a quaternion.

    Args:
        rotvec (np.ndarray): Rotation vector [vx, vy, vz].

    Returns:
        np.ndarray: Quaternion [w, x, y, z] representing the rotation.
    """

    theta = np.linalg.norm(rotvec)
    if theta < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / theta
    sin_half_theta = np.sin(theta / 2)
    cos_half_theta = np.cos(theta / 2)
    return np.array([cos_half_theta, sin_half_theta * axis[0], sin_half_theta * axis[1], sin_half_theta * axis[2]])


@jit(nopython=True)
def quaternion_to_euler(q):

    """
    Converts a quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q (np.ndarray): Quaternion [w, x, y, z] representing the rotation.

    Returns:
        np.ndarray: Euler angles [roll, pitch, yaw] in radians.
    """

    w, x, y, z = q
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    elif t2 < -1.0:
        t2 = -1.0
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return np.array([roll, pitch, yaw])

@jit(nopython=True)
def input_to_thrust(input):

    """
    Converts a control input to thrust.

    Args:
        input (float): Control input.

    Returns:
        float: Corresponding thrust.
    """

    return 2.130295e-11 * input**2 + 1.032633e-06 * input + 5.484560e-04

@jit(nopython=True)
def force_to_torque(force):

    """
    Converts a force to torque.

    Args:
        force (float): Force applied.

    Returns:
        float: Corresponding torque.
    """

    return 5.964552e-03 * force + 1.563383e-05


@jit(nopython=True)
def getGate(pos):

    """
    Determines the coordinates and angles of the gate based on the drone's position.

    Args:
        pos (np.ndarray): Drone's current position [x, y, z].

    Returns:
        np.ndarray: Gate coordinates, previous gate coordinates, angles alpha, beta, and gamma.
    """

    x = pos[0]
    y = pos[1]
    z = pos[2]


    if 2.0 >= x >= -1.0 and y <= 1.0:
        a = 2.0
        b = 0.0
        c = 1.0
        e = -1.0
        f = 0.0
        g = 1.0
        alpha = 225
        beta = 225  # degrees

    elif x >= 2.0 and y <= 1.0:
        a = 3.0
        b = 1.0
        c = 1.0
        e = 2.0
        f = 0.0
        g = 1.0
        alpha = 135
        beta = 180
    elif x >= 2.0 and y >= 1.0:
        a = 2.0
        b = 2.0
        c = 1.0
        e = 3.0
        f = 1.0
        g = 1.0
        alpha = 45
        beta = 90
    elif 2.0 >= x >= -1.0 and y >= 1.0:
        a = -1.0
        b = 3.0
        c = 1.0
        e = 2.0
        f = 2.0
        g = 1.0
        alpha = 45
        beta = 63.43494882
    elif x <= -1.0 and y >= 1.0:
        a = -3.0
        b = 1.0
        c = 1.0
        e = -1.0
        f = 3.0
        g = 1.0
        alpha = 315
        beta = 0
    elif x <= -1.0 and y <= 1.0:
        a = -1.0
        b = 0.0
        c = 1.0
        e = -3.0
        f = 1.0
        g = 1.0
        alpha_ = 225  # angle in coord system
        beta_ = 251.565  # angle to prev gate
    else:
        # print("Error ", x_, y_, z_)
        pass

    # print(x_, y_, z_)

    alpha = alpha * np.pi / 180
    beta = beta * np.pi / 180

    gate_coords = np.array([a, b, c])
    gate_prev = np.array([e,f,g])
    gate_normal_direction = np.array([np.cos(alpha) + np.sin(alpha), -np.sin(alpha) + np.cos(alpha), 0.0])
    direction_drone_to_gate = gate_coords - pos

    gamma_ = np.arccos(np.dot(gate_normal_direction, direction_drone_to_gate) / (np.linalg.norm(gate_normal_direction) * np.linalg.norm(direction_drone_to_gate)))

    #gamma_ = gamma_ * 180 / np.pi

    gate_obs = np.array([gate_coords[0], gate_coords[1], gate_coords[2], gate_prev[0], gate_prev[1], gate_prev[2], alpha_, beta_, gamma_, ])

    return gate_obs

@jit(nopython=True)
def input_to_rotvel(input):

    """
    Converts a control input to Motor Omega.

    Args:
        input (float): Control input.

    Returns:
        float: Corresponding Motor Omega.
    """

    return 4.076521e-02 * input + 380.8359


@jit(nopython=True)
def quaternion_to_rotation_matrix(quaternion):

    """

    Convert a quaternion to a rotation matrix.    
    Parameters:
    quaternion : array-like of shape (4,)
        Quaternion in the form [w, x, y, z]
    
    Returns:
    np.ndarray of shape (3, 3)
        Rotation matrix
    """

    w, x, y, z = quaternion
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R



@jit(nopython=True)
def dynamics(state, control, old_control, prev_gate, prev_gate_or, next_gate, next_gate_or, next_next_gate, next_next_gate_or, vec, vec2):

    """
    Simulates the dynamics of a quadrotor drone.

    Args:
        state (np.ndarray): Current state of the drone including position, velocity, orientation, and angular velocity.
        control (np.ndarray): Control inputs for the drone (e.g., motor speeds or thrust commands).
        old_control (np.ndarray): Previous control inputs for the drone.

    Returns:
        tuple: Updated state of the drone, total reward based on the current state and control, and a boolean indicating if the episode is done.
    """
    
    # Constants
    G = np.array([0, 0, -9.81])  # Gravity constant (m/s^2)
    THRUST_TO_WEIGHT = 2.25  # Thrust to weight ratio (N/kg)
    TIMESTEP = 0.05  # Time step for the integration and derivation (s)
    M = 0.028  # Mass of the drone (kg)
    KF = 3.16e-10  # Thrust coefficient (N/(rpm)^2)
    KM = 7.94e-12  # Moment coefficient (Nm/(rpm)^2)
    L = 0.0397  # Distance from the center to the motor (m)
    HOVER_RPM = 16000  # RPM required to hover (rpm)
    MAX_RPM = 24000  # Maximum RPM (rpm)
    AIR_DENSITY = 1.225e03  # Air density (kg/(m^3))
    DRAG_COEFFICIENT = 0.0000806428  # Drag coefficient
    REFERENCE_AREA = 0.1  # Reference area for drag calculation (m^2)
    J = np.array([[16.57, 0.83, 0.718],  # Inertia matrix (kg/m^2)
                [0.83, 16.655, 1.8], 
                [0.718, 1.8, 29.26]]) * 10e-06
    J_INV = np.linalg.inv(J)
    ROTOR_INERTIA = 0.0001  # Rotor inertia
    GOAL_STATE = np.asarray([0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    TRAIN = "racing"  # Training mode

    DRAG_COEFF = np.array([[-9.1785e-07, 0, 0],
                        [0, -9.1785e-07, 0],
                        [0, 0, -10.311e-07]])

    # Reward Coefficients
    POS_REW_COEFF = -0.02
    ORI_REW_COEFF = -0.002
    VEL_REW_COEFF = -0.0002
    RPY_RATES_REW_COEFF = -0.0002
    ACT_REW_COEFF = -0.02

    A = 10
    B = 0.25

    # Unpack the state vector
    position = state[0:3]
    position_before = position.copy()
    orientation = state[6:9]  # Position [x, y, z]
    quaternion = quaternion_from_euler(orientation[0], orientation[1], orientation[2])  # Orientation quaternion [qw, qx, qy, qz]
    velocity = state[3:6]  # Linear velocity [vx, vy, vz]
    angular_velocity = state[9:12]  # Angular rates [roll_rate, pitch_rate, yaw_rate]
    motor_speeds = control[:]

    #prev_gate, next_gate, next_next_gate = gates[0], gates[1], gates[2]
    next = False


    # Convert control signals to input values
    inputs = np.empty(4)
    for i in range(4):
        inputs[i] = (control[i] * 32750 + 32750) #*0.6

    # Calculate forces generated by the motors
    forces = np.array([input_to_thrust(inputs[0]), input_to_thrust(inputs[1]), input_to_thrust(inputs[2]), input_to_thrust(inputs[3])])
    force = np.array([0, 0, np.sum(forces)])

    # Calculate torques generated by the motors
    torques = np.array([force_to_torque(forces[0]), force_to_torque(forces[1]), force_to_torque(forces[2]), force_to_torque(forces[3])])

    #print('forces:', forces, 'torques:', torques)

    # Calculate rotational velocities
    rot_vel = np.array([input_to_rotvel(inputs[0]), input_to_rotvel(inputs[1]), input_to_rotvel(inputs[2]), input_to_rotvel(inputs[3])])

    # Calculate torque components in x, y, z directions
    torque_x = (-torques[0] - torques[1] + torques[2] + torques[3]) #* (L/np.sqrt(2))
    torque_y = (-torques[0] + torques[1] + torques[2] - torques[3]) #* (L/np.sqrt(2))
    torque_z = (-torques[0] + torques[1] - torques[2] + torques[3])
    torque = np.array([torque_x, torque_y, torque_z])

    # Calculate linear and angular accelerations

    rot_mat = quaternion_to_rotation_matrix(quaternion)
    
    drag_force = np.dot(DRAG_COEFF * np.sum(rot_vel), np.dot(np.linalg.inv(rot_mat), velocity))

    acc = (quaternion_rotate(quaternion, (force + drag_force) / M)) + G # not rot vel! # np.dot(DRAG_COEFF, velocity) * np.sum(angular_velocity)
    

    angular_acc = np.dot(J_INV, (torque - np.cross(angular_velocity, np.dot(J, angular_velocity))))

    #print('acc:', acc, 'ang_acc:', angular_acc)
    #print('drag:', drag_force)

    # Update state variables
    velocity += acc * TIMESTEP
    position += velocity * TIMESTEP
    angular_velocity += angular_acc * TIMESTEP
    #orientation += angular_velocity * TIMESTEP

    angular_velocity_quat = quaternion_from_rotvec(angular_velocity * TIMESTEP)

    orientation = quaternion_multiply(quaternion, angular_velocity_quat)

    orientation_euler = quaternion_to_euler(orientation)

    obs = np.array([position[0], position[1], position[2],
                    velocity[0], velocity[1], velocity[2],
                    orientation_euler[0], orientation_euler[1], orientation_euler[2],
                    angular_velocity[0], angular_velocity[1], angular_velocity[2]])

        
    if TRAIN == "position":

        """
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                        POINT BASED REWARD
                                                                        ------------------
                                                    Reward for being close to a GOAL_POSE in position, orientation,
                                                    velocity and angular_velocity. 
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------
        """
        
        # Compute rewards based on the state and goal state
        position_reward = POS_REW_COEFF * np.sqrt((obs[0] - GOAL_STATE[0])**2 + (obs[1] - GOAL_STATE[1])**2 + (obs[2] - GOAL_STATE[2])**2)
        orientation_reward = ORI_REW_COEFF * np.sqrt((obs[6] - GOAL_STATE[6])**2 + (obs[7] - GOAL_STATE[7])**2 + (obs[8] - GOAL_STATE[8])**2)
        velocity_reward = VEL_REW_COEFF * np.sqrt((obs[3] - GOAL_STATE[3])**2 + (obs[4] - GOAL_STATE[4])**2 + (obs[5] - GOAL_STATE[5])**2)
        rpy_rates_reward = RPY_RATES_REW_COEFF * np.sqrt((obs[9] - GOAL_STATE[9])**2 + (obs[10] - GOAL_STATE[10])**2 + (obs[11] - GOAL_STATE[11])**2)

        act_reward = ACT_REW_COEFF * np.linalg.norm(control - old_control)

        # Total reward
        total_reward = position_reward + orientation_reward + velocity_reward + act_reward + rpy_rates_reward 
        total_reward += 0.1

    elif TRAIN == "racing":

        x, y, z = position[0], position[1], position[2]
        x_b, y_b, z_b = position_before[0], position_before[1], position_before[2]

        
        gate_reward = 0
        next = False
        g2 = next_gate
        g1 = prev_gate
        good = False

        # check if through of next_gate

        if next_gate_or == 1.0:
            if x > next_gate[0] and x_b < next_gate[0]:
                next = True
                if (next_gate[1] - 0.3 < y < next_gate[1] + 0.3) and (next_gate[2] - 0.3 < z < next_gate[2] + 0.3):
                    gate_reward = 60  #/ np.linalg.norm(np.array([next_gate[0]-position[0], next_gate[1]-position[1], next_gate[2]-position[2]]))
                    good = True
                    #print("Now")
                else:
                    gate_reward = - 10*np.linalg.norm(np.array([next_gate[0]-position[0], next_gate[1]-position[1], next_gate[2]-position[2]])) 

        elif next_gate_or == -1.0:
            if x < next_gate[0] and x_b > next_gate[0]:
                next = True
                if (next_gate[1] - 0.3 < y < next_gate[1] + 0.3) and (next_gate[2] - 0.3 < z < next_gate[2] + 0.3):
                    gate_reward = 60 #/ np.linalg.norm(np.array([next_gate[0]-position[0], next_gate[1]-position[1], next_gate[2]-position[2]]))
                    good = True
                    #print("Now")
                else:
                    gate_reward = - 10* np.linalg.norm(np.array([next_gate[0]-position[0], next_gate[1]-position[1], next_gate[2]-position[2]]))

        elif next_gate_or == 2.0:
            if y > next_gate[1] and y_b < next_gate[1]:
                next = True
                if (next_gate[0] - 0.3 < x < next_gate[0] + 0.3) and (next_gate[2] - 0.3 < z < next_gate[2] + 0.3):
                    gate_reward = 60 #/ np.linalg.norm(np.array([next_gate[0]-position[0], next_gate[1]-position[1], next_gate[2]-position[2]]))
                    good = True
                    #print("Now")
                else:
                    gate_reward = - 10* np.linalg.norm(np.array([next_gate[0]-position[0], next_gate[1]-position[1], next_gate[2]-position[2]]))

        elif next_gate_or == -2.0:
            if y < next_gate[1] and y_b > next_gate[1]:
                next = True
                if (next_gate[0] - 0.3 < x < next_gate[0] + 0.3) and (next_gate[2] - 0.3 < z < next_gate[2] + 0.3):
                    gate_reward = 60 #20 / np.linalg.norm(np.array([next_gate[0]-position[0], next_gate[1]-position[1], next_gate[2]-position[2]]))
                    good = True
                    #print("Now")
                else:
                    gate_reward = - 10* np.linalg.norm(np.array([next_gate[0]-position[0], next_gate[1]-position[1], next_gate[2]-position[2]]))
        else:
            print('Error')

        if next == True: 
            g2 = next_next_gate
            g1 = next_gate

        else:
            g2 = next_gate
            g1 = prev_gate

        #if (x_b < 2.0 and y_b > -0.2 and y_b < 0.2 and z_b < 1.5 and z_b > 0.5 and x > 2.0 and y > -0.5 and y < 0.5 and z > 0.5 and z < 1.5) or (x_b > 2.8 and x_b < 3.2 and y_b < 1.0 and z_b > 0.5 and z_b < 1.5 and x > 2.5 and x < 3.5 and y > 1.0 and z > 0.5 and z < 1.5) or (x_b > 2.0 and y_b > 1.8 and y_b < 2.2 and z_b < 1.5 and z_b > 0.5 and x < 2.0 and y > 1.5 and y < 2.5 and z > 0.5 and z < 1.5) or (x_b > -1.0 and y_b > 2.8 and y_b < 3.2 and z_b < 1.5 and z_b > 0.5 and x < -1.0 and y > 2.5 and y < 3.5 and z > 0.5 and z < 1.5) or (x_b > -3.2 and x_b < -2.8 and y_b > 1.0 and z_b > 0.5 and z_b < 1.5 and x > -3.5 and x < -2.5 and y < 1.0 and z > 0.5 and z < 1.5) or (x_b < -1.0 and y_b > -0.2 and y_b < 0.2 and z_b < 1.5 and z_b > 0.5 and x > -1.0 and y > -0.5 and y < 0.5 and z > 0.5 and z < 1.5):
        #    gate_reward = 20 # with 100 2024-07-14-15-55-26_Iteration_1162.zip working great 

        #gate_obs = getGate(position)

        #g2 = gate_obs[0:3] # next gate
        #g1 = gate_obs[3:6]
        #alpha, beta, gamma = gate_obs[6], gate_obs[7], gate_obs[8]
        
        """
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                        RACING BASED REWARD
                                                                        -------------------
                                                    Reward for following a racing trajectory constitued of Goals.
                                                    Additionally implemented Safety Reward to punish for being too
                                                    far from the gate center
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------
        """

        """
        ------------------------------------------------------------------ POSITIONAL REWARD ------------------------------------------------------------------------------------------------------------------------------------------

                                                    Reward which mesures the movement along a centerline between the 
                                                    two gate centers using Dotproduct
        """

        norm_vec = np.linalg.norm(g2-g1) # Distance for Norm

        #print(position.tolist())

        reward_ = ((position[0] - g1[0])*(g2[0]-g1[0])+(position[1]-g1[1])*(g2[1]-g1[1])+(position[2]-g1[2])*(g2[2]-g1[2]))/norm_vec 
        rewardprev_ = ((position_before[0] - g1[0])*(g2[0]-g1[0])+(position_before[1]-g1[1])*(g2[1]-g1[1])+(position_before[2]-g1[2])*(g2[2]-g1[2]))/norm_vec # Dotproduct

        pos_reward = reward_ - rewardprev_

        total_reward = 10*pos_reward + 0.1 + gate_reward*2

        rel_pos_next = np.array([next_gate[0] - position[0], next_gate[1] - position[1], next_gate[2] - position[2]])
        rel_pos_next_next = np.array([next_next_gate[0]-position[0], next_next_gate[1] - position[1], next_next_gate[2] - position[2]])

        r = np.linalg.norm(rel_pos_next)
        theta = np.arctan2(rel_pos_next[1], rel_pos_next[0])
        phi = np.arccos(rel_pos_next[2]/r)

        alpha = np.arccos((np.dot(rel_pos_next, vec)/(np.linalg.norm(rel_pos_next)*np.linalg.norm(vec))))


        r2 = np.linalg.norm(rel_pos_next_next)
        theta2 = np.arctan2(rel_pos_next_next[1], rel_pos_next_next[0])
        phi2 = np.arccos(rel_pos_next_next[2]/r2)

        alpha2 = np.arccos((np.dot(rel_pos_next_next, vec2)/(np.linalg.norm(rel_pos_next_next)*np.linalg.norm(vec2))))

        
        

            
        
        """
        ------------------------------------------------------------------ SAFETY REWARD ------------------------------------------------------------------------------------------------------------------------------------------------

                                                    Reward which mesures the distance to the gate center (safety reasons)

        """
        
        """gate_normal_direction = np.array([np.cos(alpha) + np.sin(alpha), - np.sin(alpha) + np.cos(alpha), 0]) 

        D_ = - (gate_normal_direction[0]*g2[0] + gate_normal_direction[1]*g2[1] + gate_normal_direction[2]*g2[2])
        plane_of_gate = np.array([gate_normal_direction[0], gate_normal_direction[1], gate_normal_direction[2], D_]) 

        gate_side_length = 1 # meter
        distance_max = 1 # meter
        distance_plane = abs((g2[0]*plane_of_gate[0] + g2[1]*plane_of_gate[1] + g2[2]*plane_of_gate[2] + plane_of_gate[3])/(np.sqrt(plane_of_gate[0]**2 + plane_of_gate[1]**2 + plane_of_gate[2]**2))) 

            
        projection = ((gate_normal_direction[0]*position[0] + gate_normal_direction[1]*position[1] + gate_normal_direction[2]*position[2] - gate_normal_direction[0]*g2[0] + gate_normal_direction[1]*g2[1] - gate_normal_direction[2]*g2[2]) 
                        /(gate_normal_direction[0]**2 + gate_normal_direction[1]**2 + gate_normal_direction[2]**2)) * gate_normal_direction + g2 
            
        distance_to_normal = np.linalg.norm(position - projection)

        f = 1 - (distance_plane/distance_max) # Safety Reward only if drone is close enough!

        if (f < 0): 
            f = 0

        v = (1 - f)*(gate_side_length/5)
            
        if (v < 0.05): 
            v = 0.05

        safety_reward = -f * f * (1 - np.exp((-0.5 * distance_to_normal * distance_to_normal)/v))

        total_reward = A * pos_reward + gate_reward #+ 0.1 * safety_reward 

        total_reward += 0.2 

        total_reward /= 5 # 10
        """
        
        #print(gate_reward)
    
    done = False
    """if (position[0] > 3.5 or position[0] < -3.5 or position[1] < -0.5 or position[1] > 3.5 or 
        position[2] > 1.6 or position[2] < 1.0 or angular_velocity[0] > 100 or angular_velocity[1] > 100 or 
        angular_velocity[2] > 100 or angular_velocity[0] < -100 or angular_velocity[1] < -100 or angular_velocity[2] < -100):
        done = True"""

    if (position[0] > 6.1 or position[0] < -6.1 or position[1] < -6.1 or position[1] > 6.1 or 
        position[2] > 4 or position[2] < -1 or angular_velocity[0] > 100 or angular_velocity[1] > 100 or 
        angular_velocity[2] > 100 or angular_velocity[0] < -100 or angular_velocity[1] < -100 or angular_velocity[2] < -100):
        done = True
        #print(obs)


    
    obs = np.array([position[0], position[1], position[2],
                    velocity[0], velocity[1], velocity[2],
                    orientation_euler[0], orientation_euler[1], orientation_euler[2],
                    angular_velocity[0], angular_velocity[1], angular_velocity[2],
                    r, theta, phi,
                    alpha,
                    #r2, theta2, phi2, alpha2
                    ])
    


    #print(control)
    # Control Outputs

    #print(position_reward, orientation_reward, velocity_reward, rpy_rates_reward)
    #print('Angular_momentum:', angular_momentum)
    #print('Angular_momentum_change:', angular_momentum_change)
    #print('Tau Control:', tau_control)
    #print(position, quaternion, velocity, rpy_rates)

    return obs, total_reward, done, next, good

