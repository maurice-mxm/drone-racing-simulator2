import numpy as np
from numba import jit

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                        drone_dynamics.py
                                                                        -----------------
                                        This is the main Python File, where all of the simulation is done. This is done
                                        by solving 6 DoF Equations to get the state of the quadcopter after a certain
                                        input
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


@jit(nopython=True)
def getMatrixFromQuaternion(quat):

    """
    Convert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param quat: A 4 element array representing the quaternion (q0, q1, q2, q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """

    # Extract the values from Q
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]
     
    # First row of the rotation matrix
    r00 = -2 * (q0 * q0 + q1 * q1) + 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = -2 * (q1 * q1 + q3 * q3) + 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = -2 * (q1 * q1 + q2 * q2) + 1 # vorher q3 * q3
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix


@jit(nopython=True)
def integrateQ(quat, omega, dt):

    """
    Integrate the quaternion based on angular velocity and time step.

    Input
    :param quat: A 4 element array representing the current quaternion (q0, q1, q2, q3)
    :param omega: A 3 element array representing the angular velocity (p, q, r)
    :param dt: A float representing the time step for integration

    Output
    :return: A 4 element array representing the updated quaternion after integration
    """

    omega_norm = np.linalg.norm(omega)
    if omega_norm < 1e-10:
        return quat
    p, q, r = omega
    lambda_ = np.array([
        [0, r, -q, p],
        [-r, 0, p, q],
        [q, -p, 0, r],
        [-p, -q, -r, 0]
    ]) * 0.5
    theta = omega_norm * dt / 2
    quat = np.dot(np.eye(4) * np.cos(theta) + 2 / omega_norm * lambda_ * np.sin(theta), quat)
    return quat

    """omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    quat_dot = 0.5 * quaternion_multiply(quat, omega_quat)
    quat = quat + quat_dot * dt
    return quat / np.linalg.norm(quat)"""


@jit(nopython=True)
def quaternion_multiply(q, r):
    """
    Multiplies two quaternions q and r.
    
    Input
    :param q: A 4 element array representing the first quaternion (q0, q1, q2, q3)
    :param r: A 4 element array representing the second quaternion (r0, r1, r2, r3)
    
    Output
    :return: A 4 element array representing the product quaternion
    """
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
    ])


@jit(nopython=True)
def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.

    Input
    :param roll: The roll angle (rotation around x-axis) in radians
    :param pitch: The pitch angle (rotation around y-axis) in radians
    :param yaw: The yaw angle (rotation around z-axis) in radians

    Output
    :return: A 4 element array representing the quaternion (q0, q1, q2, q3)
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy

    return np.array([q0, q1, q2, q3])

# Example usage:
roll = np.radians(30)   # Convert 30 degrees to radians
pitch = np.radians(45)  # Convert 45 degrees to radians
yaw = np.radians(60)    # Convert 60 degrees to radians

quat = euler_to_quaternion(roll, pitch, yaw)
print("Quaternion:", quat)


@jit(nopython=True)
def rotation_matrix(phi, theta, psi):

    """
    Compute the rotation matrix from roll, pitch, and yaw angles. Currently !not! used.

    Input
    :param phi: A float representing the roll angle
    :param theta: A float representing the pitch angle
    :param psi: A float representing the yaw angle

    Output
    :return: A 3x3 element matrix representing the rotation matrix
    """

    cphi = np.cos(phi)
    sphi = np.sin(phi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    
    R = np.empty((3, 3))
    R[0, 0] = cphi * cpsi - ctheta * sphi * spsi
    R[0, 1] = -ctheta * cpsi * sphi + cphi * spsi
    R[0, 2] = sphi * stheta
    R[1, 0] = -cpsi * sphi - cphi * ctheta * spsi
    R[1, 1] = cphi * ctheta * cpsi - sphi * spsi
    R[1, 2] = cphi * stheta
    R[2, 0] = stheta * spsi
    R[2, 1] = -cpsi * stheta
    R[2, 2] = ctheta

    return R
    
@jit(nopython=True)
def dynamics(state, control, old_control, quat):

    """
    Calculate the next state of the drone based on current state, control inputs, and dynamics.

    Input
    :param state: A 13 element array representing the current state of the drone:
                  [position_x, position_y, position_z, quaternion_w, quaternion_x, quaternion_y, quaternion_z,
                   velocity_x, velocity_y, velocity_z, roll_rate, pitch_rate, yaw_rate]
    :param control: A 4 element array representing the current control inputs (motor speeds)
    :param old_control: A 4 element array representing the previous control inputs (motor speeds)

    Output
    :return: A tuple containing:
             - obs: A 13 element array representing the updated state of the drone
             - total_reward: A float representing the reward based on the state transition / current state
             - done: A boolean indicating if the drone is in an invalid state or if the environment has to be resetted
    """
    
    # Constants and parameters
    G = 9.81  # Gravity constant (m/s^2)
    THRUST_TO_WEIGHT = 2.25  # Thrust to weight ratio (N/kg)
    TIMESTEP = 0.02  # Time step for the integration and deriviation (s)
    M = 0.027  # Mass of the drone (kg)
    GRAVITY = G * M  # Gravity force acting on the drone (N)
    KF = 3.16e-10  # Thrust coefficient (N/(rpm)^2)
    KM = 7.94e-12  # Moment coefficient (Nm/(rpm)^2)
    L = 0.0397  # Distance from the center to the motor (m)
    HOVER_RPM = 16000  # RPM required to hover (rpm)
    MAX_RPM = 24000  # Maximum RPM (rpm)
    AIR_DENSITY = 1.225e03  # Air density (kg/(m^3))
    DRAG_COEFFICIENT = 0.0000806428  # Drag coefficient (...)
    REFERENCE_AREA = 0.1  # Reference area for drag calculation (m^2)
    J = np.array([[1.4e-05, 0, 0],  # Inertia matrix (kg/m^2)
                  [0, 1.4e-05, 0], 
                  [0, 0, 2.17e-05]])
    J_INV = np.array([[1/0.0000145, 0, 0],  # Inverse of the inertia matrix (m^2/kg) ...
                      [0, 1/0.0000145, 0], 
                      [0, 0, 1/0.0000217]])
    ROTOR_INERTIA = 0.0001  # Rotor inertia
    GOAL_STATE = np.asarray([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    TRAIN = "position" # or "racing"

    DRAG_COEFF = 9.1785E-07

    # Reward Coefficients:
    POS_REW_COEFF = -0.02
    ORI_REW_COEFF = -0.002
    VEL_REW_COEFF = -0.0002
    RPY_RATES_REW_COEFF = -0.00002
    ACT_REW_COEFF = -0.02

    A = 10
    B = 0.25

    # Unpack the state vector
    position = state[0:3]
    orientation = state[6:9]  # Position [x, y, z]
    quaternion = quat[:]  # Orientation quaternion [qw, qx, qy, qz]
    velocity = state[3:6]  # Linear velocity [vx, vy, vz]
    rpy_rates = state[9:12]  # Angular rates [roll_rate, pitch_rate, yaw_rate]

    # Calculate the rotation matrix from the quaternion
    rotation = getMatrixFromQuaternion(quaternion)

    # Compute the RPM for each motor based on the control inputs
    rpm = np.where(control <= 0, (control + 1) * HOVER_RPM, HOVER_RPM + (MAX_RPM - HOVER_RPM) * control)

    # Compute the forces generated by each motor
    forces = np.array([rpm[0]**2 * KF, rpm[1]**2 * KF, rpm[2]**2 * KF, rpm[3]**2 * KF])

    # Compute the total thrust force in the body frame
    thrust = np.array([0, 0, np.sum(forces)])

    # Convert the thrust force to the world frame
    thrust_world_frame = np.dot(rotation, thrust)

    # Compute drag generated
    drag_factor = -1 * DRAG_COEFF * np.sum(2*np.pi*rpm/60)
    drag_force = np.dot(rotation, drag_factor*velocity)

    # Calculate the total force acting on the drone in the world frame
    force_world_frame = thrust_world_frame + drag_force - np.array([0, 0, GRAVITY])

    # Compute the torques generated by each motor
    z_torques = np.array([rpm[0]**2 * KM, rpm[1]**2 * KM, rpm[2]**2 * KM, rpm[3]**2 * KM])
    z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
    x_torque = (forces[1] - forces[3]) * L / np.sqrt(2)
    y_torque = (-forces[0] + forces[2]) * L / np.sqrt(2)

    # Total torque vector
    torques = np.array([x_torque, y_torque, z_torque])

    # Apply the gyroscopic effect correction
    torques = torques - np.cross(rpy_rates, np.dot(J, rpy_rates))

    # Calculate the angular acceleration
    rpy_rates_deriv = np.dot(J_INV, torques)

    # Calculate the linear acceleration
    no_pybullet_dyn_accs = force_world_frame / M

    # Integrate to get new state values
    velocity = velocity + TIMESTEP * no_pybullet_dyn_accs
    rpy_rates = rpy_rates + TIMESTEP * rpy_rates_deriv
    position = position + TIMESTEP * velocity
    orientation = orientation + TIMESTEP * rpy_rates

    quaternion = integrateQ(quaternion, rpy_rates, TIMESTEP)

    # Construct the new state observation
    obs = np.array([position[0], position[1], position[2],
                    velocity[0], velocity[1], velocity[2],
                    orientation[0], orientation[1], orientation[2],
                    rpy_rates[0], rpy_rates[1], rpy_rates[2]])
    
    
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
        total_reward = position_reward + orientation_reward + velocity_reward + rpy_rates_reward + act_reward
        total_reward += 0.15

    elif TRAIN == "racing":
        
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

        reward_ = ((pos_now[0] - g1[0])*(g2[0]-g1[0])+(pos_now[1]-g1[1])*(g2[1]-g1[1])+(pos_now[2]-g1[2])*(g2[2]-g1[2]))/norm_vec 
        rewardprev_ = ((pos_before[0] - g1[0])*(g2[0]-g1[0])+(pos_before[1]-g1[1])*(g2[1]-g1[1])+(pos_before[2]-g1[2])*(g2[2]-g1[2]))/norm_vec # Dotproduct

        pos_reward = reward_ - rewardprev_

            

        """
        ------------------------------------------------------------------ SAFETY REWARD ------------------------------------------------------------------------------------------------------------------------------------------------

                                                    Reward which mesures the distance to the gate center (safety reasons)

        """

        gate_normal_direction = np.array(math.cos(alpha) + math.sin(alpha), - math.sin(alpha) + math.cos(alpha), 0) 

        D_ = - (gate_normal_direction[0]*g2[0] + gate_normal_direction[1]*g2[1] + gate_normal_direction[2]*g2[2])
        plane_of_gate = np.array(gate_normal_direction[0], gate_normal_direction[1], gate_normal_direction[2], D_) 

        gate_side_length = 1 # meter
        distance_max = 1 # meter
        distance_plane = abs((g2[0]*plane_of_gate[0] + g2[1]*plane_of_gate[1] + g2[2]*plane_of_gate[2] + plane_of_gate[3])/(math.sqrt(plane_of_gate[0]^2 + plane_of_gate[1]^2 + plane_of_gate[2]^2))) 

            
        projection = ((gate_normal_direction[0]*pos_now[0] + gate_normal_direction[1]*pos_now[1] + gate_normal_direction[2]*pos_now[2] - gate_normal_direction[0]*g2[0] + gate_normal_direction[1]*g2[1] - gate_normal_direction[2]*g2[2]) 
                        /(gate_normal_direction[0]^2 + gate_normal_direction[1]^2 + gate_normal_direction[2]^2)) * gate_normal_direction + g2 
            
        distance_to_normal = np.linalg.norm(pos_now - projection)

        f = 1 - (distance_plane/distance_max) # Safety Reward only if drone is close enough!

        if (f < 0): 
            f = 0

        v = (1 - f)*(gate_side_length/5)
            
        if (v < 0.05): 
            v = 0.05

        safety_reward = -f * f * (1 - np.exp((-0.5 * distance_to_normal * distance_to_normal)/v))

        total_reward = A * pos_reward + B * safety_reward

    #print(position, rpy_rates)

    # Check if the state is invalid
    done = False
    if (position[0] > 6 or position[0] < -6 or position[1] > 6 or position[1] < -6 or 
        position[2] > 8 or position[2] < 0.2 or rpy_rates[0] > 100 or rpy_rates[1] > 100 or 
        rpy_rates[2] > 100 or rpy_rates[0] < -100 or rpy_rates[1] < -100 or rpy_rates[2] < -100):
        done = True

    #print(control)
    # Control Outputs

    #print(position_reward, orientation_reward, velocity_reward, rpy_rates_reward)
    #print('Angular_momentum:', angular_momentum)
    #print('Angular_momentum_change:', angular_momentum_change)
    #print('Tau Control:', tau_control)
    #print(position, quaternion, velocity, rpy_rates)

    return obs, total_reward, done, quaternion

"""
Test for correct Configuration (not used right now!)
"""

"""
initial_state = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
timeline = []
state = initial_state

for i in range(100000):
    if i == 5:
        begin = time.time()
    l = [random.randint(1, 10000), random.randint(1, 10000), random.randint(1, 10000), random.randint(1, 10000)]
    time_start = time.time()
    state = dynamics(state, l, 0.01)
    time_end = time.time()
    timeline.append(time_end - time_start)

end = time.time()
print(timeline)
print(end - begin)
"""
