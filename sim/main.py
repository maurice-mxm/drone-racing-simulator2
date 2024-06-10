import numpy as np
from numba import jit
import time
import random
import math

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
def dynamics(state, control):


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
