import numpy as np
from numba import jit
import time
import random
import math



@jit(nopython=True)
def getMatrixFromQuaternion(quat):

    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
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
    r22 = -2 * (q1 * q1 + q3 * q3) + 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

@jit(nopython=True)
def integrateQ(quat, omega, dt):
    omega_norm = np.linalg.norm(omega)
    p, q, r = omega
    #if 0.1 > omega_norm > 0.0: #if np.isclose(omega_norm, 0):
    #    return quat
    lambda_ = np.array([
        [ 0,  r, -q, p],
        [-r,  0,  p, q],
        [ q, -p,  0, r],
        [-p, -q, -r, 0]
    ]) * 0.5
    theta = omega_norm * dt / 2
    quat = np.dot(np.eye(4) * np.cos(theta) + 2 / omega_norm * lambda_ * np.sin(theta), quat)
    return quat


@jit(nopython=True)
def rotation_matrix(phi, theta, psi):
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
        
    R = np.empty((3, 3))
    """
    R[0, 0] = cpsi * ctheta
    R[0, 1] = cpsi * stheta * sphi - spsi * cphi
    R[0, 2] = cpsi * stheta * cphi + spsi * sphi
    R[1, 0] = spsi * ctheta
    R[1, 1] = spsi * stheta * sphi + cpsi * cphi
    R[1, 2] = spsi * stheta * cphi - cpsi * sphi
    R[2, 0] = -stheta
    R[2, 1] = ctheta * sphi
    R[2, 2] = ctheta * cphi
    """
    R[0, 0] = cphi*cpsi - ctheta*sphi*spsi
    R[0, 1] = -ctheta*cpsi*sphi + cphi*spsi
    R[0, 2] = sphi*stheta
    R[1, 0] = -cpsi*sphi - cphi*ctheta*spsi
    R[1, 1] = cphi*ctheta*cpsi - sphi*spsi
    R[1, 2] = cphi*stheta
    R[2, 0] = stheta*spsi
    R[2, 1] = -cpsi*stheta
    R[2, 2] = ctheta

    return R
    
@jit(nopython=True)
def dynamics(state, control, old_control):

    #np.seterr(all='raise')
    G = 9.81
    
    thrust_to_weight = 2.25

    time_step = 0.02
    m = 0.027 # 1.5
    GRAVITY = G * m
    kF = 3.16e-10 #8.5e-06 #1e-5
    kM = 7.94e-12 #1e-06
    l = 0.0397 #0.14
    HOVER_RPM = 16000 # 650
    MAX_RPM = 24000 #1100
    air_density = 1.225
    drag_coefficient = 0.0000806428
    reference_area = 0.1
    J = np.array([[1.4e-05, 0, 0], # 0.0291
                 [0, 1.4e-05, 0], # 0.0291
                 [0, 0, 2.17e-05]]) # 0.055
    
    J_INV = np.array([[1/0.0000145, 0, 0],
                 [0, 1/0.0000145, 0], 
                 [0, 0, 1/0.0000217]])

    
    rotor_inertia = 0.0001
        
    position = state[0:3]
    quaternion = state[3:7]
    velocity = state[7:10]
    rpy_rates = state[10:13]

    rotation = getMatrixFromQuaternion(quaternion)
    #print(quaternion)

    #phi, theta, psi = orientation

    # Position derivatives
    rpm = np.where(control <= 0, (control+1)*HOVER_RPM, HOVER_RPM + (MAX_RPM - HOVER_RPM)*control) #np.empty(4)
    #print(rpm)
    #for i in range(4):
    #        rpm[i] = 1000*control[i]

    forces = np.array([rpm[0]**2 * kF, rpm[1]**2 * kF, rpm[2]**2 * kF, rpm[3]**2 * kF]) # np.array(rpm**2) * kF
        
    thrust = np.array([0, 0, np.sum(forces)])
    thrust_world_frame = np.dot(rotation, thrust)
    force_world_frame = thrust_world_frame - np.array([0, 0, GRAVITY])
    z_torques = np.array([rpm[0]**2 * kM, rpm[1]**2 * kM, rpm[2]**2 * kM, rpm[3]**2 * kM]) # np.array(rpm**2) * kM

    z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])

    x_torque = (forces[1] - forces[3]) * l/np.sqrt(2)
    y_torque = (-forces[0] + forces[2]) * l/np.sqrt(2)
    

    torques = np.array([x_torque, y_torque, z_torque])
    #print(force_world_frame)

    torques = torques - np.cross(rpy_rates, np.dot(J, rpy_rates))
    rpy_rates_deriv = np.dot(J_INV, torques)

    

    no_pybullet_dyn_accs = force_world_frame / m

    velocity = velocity + time_step * no_pybullet_dyn_accs
    rpy_rates = rpy_rates + time_step * rpy_rates_deriv
    position = position + time_step * velocity
    quaternion = integrateQ(quaternion, rpy_rates, time_step)

    #print(position[1])




    

    obs = np.array([position[0], position[1], position[2], 
                quaternion[0], quaternion[1], quaternion[2], quaternion[3], 
                velocity[0], velocity[1], velocity[2],
                rpy_rates[0], rpy_rates[1], rpy_rates[2]])
    
    goal_state = np.asarray([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    #act_reward = -0.02 * np.sqrt((old_control[0]-control[0])**2 + (old_control[1]-control[1])**2 + (old_control[2]-control[2])**2 + (old_control[3]-control[3])**2) 
    
    position_reward = -0.002 * np.sqrt((obs[0]-goal_state[0])**2 + (obs[1]-goal_state[1])**2 + (obs[2]-goal_state[2])**2)
    orientation_reward = -0.002 * np.sqrt((obs[3]-goal_state[3])**2 + (obs[4]-goal_state[4])**2 + (obs[5]-goal_state[5])**2 + (obs[6]-goal_state[6])**2)
    velocity_reward = -0.0002 * np.sqrt((obs[7]-goal_state[7])**2 + (obs[8]-goal_state[8])**2 + (obs[9]-goal_state[9])**2)
    rpy_rates_reward = -0.0002 * np.sqrt((obs[10]-goal_state[10])**2 + (obs[11]-goal_state[11])**2 + (obs[12]-goal_state[12])**2)

    total_reward = position_reward + orientation_reward + velocity_reward + rpy_rates_reward #+ act_reward

    total_reward += 0.1

    x = obs[0]
    y = obs[1]
    z = obs[2]
    r = obs[10]
    p = obs[11]
    y = obs[12]

    done = False

    #print(x, y, z, r, p, y)
    #print(position[1])
    #print(y)

    if position[0] > 3 or position[0] < -3 or position[1] > 3 or position[1] < -3 or position[2] > 6 or position[2] < 0.2 or rpy_rates[0] > 6 or rpy_rates[1] > 6 or rpy_rates[2] > 6 or rpy_rates[1] < -6 or rpy_rates[2] < -6 or rpy_rates[2] < -6:

        #print(position[0], position[1], position[2], rpy_rates[0], rpy_rates[1], rpy_rates[2])
        done = True



    #print(obs[10:13])

    #print(position_reward, orientation_reward, velocity_reward, rpy_rates_reward)
    #print('Angular_momentum:', angular_momentum)
    #print('Angular_momentum_change:', angular_momentum_change)
    #print('Tau Control:', tau_control)

    #print(position, quaternion, velocity, rpy_rates)

    return obs, total_reward, done
    """
    def dynamics(state, control):

    GRAVITY = 9.81
    


    time_step = 0.02
    m = 1.5
    g = 9.81
    kF = 8.4e-05 #1e-5
    kM = 0.06
    l = 0.14
    air_density = 1.225
    drag_coefficient = 0.0000806428
    reference_area = 0.1
    inertia = np.array([0.029, 0.029, 0.055])
    rotor_inertia = 0.0001
        
    position = state[0:3]
    velocity = state[3:6]
    angular_velocity = state[6:9]
    orientation = state[9:12]

    phi, theta, psi = orientation

    # Position derivatives
    ctrl = np.empty(4)
    for i in range(4):
            ctrl[i] = 100*control[i]+100

    

    # Thrust force
    sq = np.empty(4)
    for i in range(4):
        sq[i] = ctrl[i]**2
        
    thrust = np.array([0, 0, kF * np.sum(sq)])
    gravity = np.array([0, 0, -g])

    #print(thrust)

    #print(ctrl[0], ctrl[1], ctrl[2], ctrl[3])
    
    drag = np.empty(3)
    for i in range(3):
        drag[i] = -0.5 * air_density * drag_coefficient * reference_area * velocity[i]**2

    R = rotation_matrix(phi, theta, psi)
    #thrust_inertial = np.empty(3)
    #print(R)

    thrust_inertial = np.dot(R, thrust)

    #print('thrust_inertial: ', thrust_inertial[0], thrust_inertial[1], thrust_inertial[2])

    accel = gravity + (thrust_inertial) / m

    for i in range(3):
        velocity[i] += accel[i] * time_step
        position[i] += velocity[i] * time_step

    # Angular velocity derivatives
    tau_control = np.array([
        l * kF * (ctrl[0] - ctrl[2]), # changed here sq --> ctrl
        l * kF * (ctrl[1] - ctrl[3]),
        kM * (ctrl[0] - ctrl[1] + ctrl[2] - ctrl[3])
    ])

    rotor_speeds = np.empty(4)
    for i in range(4):
        rotor_speeds[i] = ctrl[i]


    omega_r = np.array([0, 0, np.sum(rotor_speeds) * kM])
    #tau_gyro = rotor_inertia * np.cross(angular_velocity, omega_r)

    angular_momentum = np.empty(3)
    for i in range(3):
        angular_momentum[i] = inertia[i] * angular_velocity[i]
    
    #np.dot(inertia, angular_velocity, angular_momentum)


    angular_momentum_change = tau_control - np.cross(angular_velocity, angular_momentum) # intertia * angular_velocity

    for i in range(3):
        angular_velocity[i] += angular_momentum_change[i] / inertia[i] * time_step




    orientation_dot = np.array([
        angular_velocity[0] + np.sin(phi) * np.tan(theta) * angular_velocity[1] + np.cos(phi) * np.tan(theta) * angular_velocity[2],
        np.cos(phi) * angular_velocity[1] - np.sin(phi) * angular_velocity[2],
        (np.sin(phi) / np.cos(theta)) * angular_velocity[1] + (np.cos(phi) / np.cos(theta)) * angular_velocity[2]
    ])

    for i in range(3):
        orientation[i] += orientation_dot[i] * time_step

    obs = np.array([position[0], position[1], position[2], velocity[0], velocity[1], velocity[2], 
                orientation[0], orientation[1], orientation[2], angular_velocity[0], angular_velocity[1], angular_velocity[2]])
    
    goal_state = np.asarray([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    position_reward = -0.002 * np.sqrt((obs[0]-goal_state[0])**2 + (obs[1]-goal_state[1])**2 + (obs[2]-goal_state[2])**2)
    orientation_reward = -0.002 * np.sqrt((obs[3]-goal_state[3])**2 + (obs[4]-goal_state[4])**2 + (obs[5]-goal_state[5])**2)
    velocity_reward = -0.0002 * np.sqrt((obs[6]-goal_state[6])**2 + (obs[7]-goal_state[7])**2 + (obs[8]-goal_state[8])**2)
    angular_velocity_reward = -0.0002 * np.sqrt((obs[9]-goal_state[9])**2 + (obs[10]-goal_state[10])**2 + (obs[11]-goal_state[11])**2)

    total_reward = position_reward + orientation_reward + velocity_reward + angular_velocity_reward

    total_reward += 0.2

    #print('Angular_momentum:', angular_momentum)
    #print('Angular_momentum_change:', angular_momentum_change)
    #print('Tau Control:', tau_control)



    return obs, total_reward
    """
    """m = 1.5
    g = 9.81
    kF = 1e-5
    kM = 1e-7
    l = 0.25
    air_density = 1.225
    drag_coefficient = 0.47
    reference_area = 0.1
    inertia = np.array([0.02, 0.02, 0.04])
    rotor_inertia = 0.01
    time_step = 0.05
        
    position = state[0:3]
    velocity = state[3:6]
    angular_velocity = state[6:9]
    orientation = state[9:12]

    phi, theta, psi = orientation

    ctrl = np.empty(4)
    for i in range(4):
        if control[i] < 0:
            ctrl[i] = 0.0
        else:
            ctrl[i] = 1000*control[i]

    # Position derivatives

    # Thrust force
    sq = np.empty(4)
    for i in range(4):
        sq[i] = ctrl[i]**2
        
    thrust = np.array([0, 0, kF * np.sum(sq)])
    gravity = np.array([0, 0, -g])
    
    drag = np.empty(3)
    for i in range(3):
        drag[i] = -0.5 * air_density * drag_coefficient * reference_area * velocity[i]**2

    R = rotation_matrix(phi, theta, psi)
    thrust_inertial = np.dot(R, thrust)

    accel = gravity + (thrust_inertial + drag) / m

    for i in range(3):
        velocity[i] += accel[i] * time_step
        position[i] += velocity[i] * time_step

    # Angular velocity derivatives
    tau_control = np.array([
        l * kF * (sq[0] - sq[2]),
        l * kF * (sq[1] - sq[3]),
        kM * (sq[0] - sq[1] + sq[2] - sq[3])
    ])

    rotor_speeds = np.empty(4)
    for i in range(4):
        rotor_speeds[i] = control[i]

    omega_r = np.array([0, 0, np.sum(rotor_speeds) * kM])
    tau_gyro = rotor_inertia * np.cross(angular_velocity, omega_r)

    angular_momentum = np.empty(3)
    for i in range(3):
        angular_momentum[i] = inertia[i] * angular_velocity[i]
    
    #np.dot(inertia, angular_velocity, angular_momentum)


    angular_momentum_change = tau_control - np.cross(angular_velocity, angular_momentum) # intertia * angular_velocity

    for i in range(3):
        angular_velocity[i] += angular_momentum_change[i] / inertia[i] * time_step

    orientation_dot = np.array([
        angular_velocity[0] + np.sin(phi) * np.tan(theta) * angular_velocity[1] + np.cos(phi) * np.tan(theta) * angular_velocity[2],
        np.cos(phi) * angular_velocity[1] - np.sin(phi) * angular_velocity[2],
        (np.sin(phi) / np.cos(theta)) * angular_velocity[1] + (np.cos(phi) / np.cos(theta)) * angular_velocity[2]
    ])

    for i in range(3):
        orientation[i] += orientation_dot[i] * time_step

    obs = np.array([position[0], position[1], position[2], velocity[0], velocity[1], velocity[2], 
                orientation[0], orientation[1], orientation[2], angular_velocity[0], angular_velocity[1], angular_velocity[2]])
    
    goal_state = np.asarray([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    position_reward = -0.002 * np.sqrt((obs[0]-goal_state[0])**2 + (obs[1]-goal_state[1])**2 + (obs[2]-goal_state[2])**2)
    orientation_reward = -0.002 * np.sqrt((obs[3]-goal_state[3])**2 + (obs[4]-goal_state[4])**2 + (obs[5]-goal_state[5])**2)
    velocity_reward = -0.0002 * np.sqrt((obs[6]-goal_state[6])**2 + (obs[7]-goal_state[7])**2 + (obs[8]-goal_state[8])**2)
    angular_velocity_reward = -0.0002 * np.sqrt((obs[9]-goal_state[9])**2 + (obs[10]-goal_state[10])**2 + (obs[11]-goal_state[11])**2)

    total_reward = position_reward + orientation_reward + velocity_reward + angular_velocity_reward

    total_reward += 0.2

    return obs, total_reward"""





"""initial_state = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
print(end - begin)"""
