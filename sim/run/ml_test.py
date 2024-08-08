import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import uav_trajectory

"""
---------------------------------------------------------------------------------------------------------------------------------------------------
                                                                ml_test.py
                                                                ----------
                                            Test to see how good the current model is. 
---------------------------------------------------------------------------------------------------------------------------------------------------
"""



def test(env, model):
    
    f3 = plt.figure()
    ax3 = plt.axes(projection='3d')
    x_values = []
    y_values = []
    z_values = []
    ex_values = []
    ey_values = []
    ez_values = []

    t = 0
    
    fig = plt.figure(figsize=(18, 12), tight_layout=True)
    gs = gridspec.GridSpec(5, 12)
    
    ax_x = fig.add_subplot(gs[0, 0:4])
    ax_y = fig.add_subplot(gs[0, 4:8])
    ax_z = fig.add_subplot(gs[0, 8:12])
    
    ax_dx = fig.add_subplot(gs[1, 0:4])
    ax_dy = fig.add_subplot(gs[1, 4:8])
    ax_dz = fig.add_subplot(gs[1, 8:12])
    
    ax_euler_x = fig.add_subplot(gs[2, 0:4])
    ax_euler_y = fig.add_subplot(gs[2, 4:8])
    ax_euler_z = fig.add_subplot(gs[2, 8:12])
    
    ax_euler_vx = fig.add_subplot(gs[3, 0:4])
    ax_euler_vy = fig.add_subplot(gs[3, 4:8])
    ax_euler_vz = fig.add_subplot(gs[3, 8:12])
    
    ax_action0 = fig.add_subplot(gs[4, 0:3])
    ax_action1 = fig.add_subplot(gs[4, 3:6])
    ax_action2 = fig.add_subplot(gs[4, 6:9])
    ax_action3 = fig.add_subplot(gs[4, 9:12])

    max_ep_length = 2000 #env.max_episode_steps
    num_rollouts = 1
    TIMESTEP = 0.1

    for n_roll in range(num_rollouts):
        pos, vel, accel, yerk, euler, ang_vel = [], [], [], [], [], []
        actions = []
        position, obs= env.reset()
        done, ep_len = False, 0

        print(obs)

        while not (done or (ep_len >= max_ep_length)):

            act, _ = model.predict(obs, deterministic=True)
            position, obs, rew, done, infos = env.step(act)

            #print(rew, ep_len, act)

            ep_len += 1
            if n_roll == 0:
                x_values.append(position[0][0])
                y_values.append(position[0][1])
                z_values.append(position[0][2])
                ex_values.append(obs[0][6])
                ey_values.append(obs[0][7])
                ez_values.append(obs[0][8])
                t += 1

            pos.append(obs[0, 0:3].tolist())
            vel.append(obs[0, 3:6].tolist())

            vel = np.asarray(vel)
            
            if ep_len == 1:
                accel.append((vel[-1]/TIMESTEP).tolist())
                accel = np.asarray(accel)

                yerk.append((accel[-1]/TIMESTEP).tolist())
            else:
                accel.append(((vel[-2]-vel[-1])/TIMESTEP).tolist())

                accel = np.asarray(accel)
                yerk.append(((accel[-2]-accel[-1])/TIMESTEP).tolist())

            vel = vel.tolist()
            
            accel = accel.tolist()            

            euler.append(obs[0, 6:9].tolist())
            ang_vel.append(obs[0, 9:12].tolist())

            actions.append(act[0, :].tolist())

        pos = np.asarray(pos)
        vel = np.asarray(vel)
        euler = np.asarray(euler)
        ang_vel = np.asarray(ang_vel)
        actions = np.asarray(actions)

        vels = np.empty(len(vel))
        accels = np.empty(len(vel))
        yerks = np.empty(len(vel))

        for i in range(len(vel)):
            vels[i] = np.linalg.norm(np.array([vel[i][0], vel[i][1]]))
            accels[i] = np.linalg.norm(np.array([accel[i][0], accel[i][1]]))
            yerks[i] = np.linalg.norm(np.array([yerk[i][0], yerk[i][1]]))

        print('MAX VEL: ', np.max(vels), '-- MIN VEL: ', np.min(vels))
        #print('MAX X-VEL: ', np.max(np.absolute(vel[:][0])), '-- MAX Y-VEL: ', np.max(np.absolute(vel[:][1])))
        print('MAX ACCEL: ',np.max(accels), '-- MIN ACCEL: ', np.min(accels))
        print('MAX YERK: ', np.max(yerks), '-- MIN YERK: ', np.min(yerks))        

        t = np.arange(0, pos.shape[0])
        ax_x.step(t, pos[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_y.step(t, pos[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_z.step(t, pos[:, 2], color="C{0}".format(
            n_roll), label="pos [x, y, z] -- trail: {0}".format(n_roll))
        
        ax_dx.step(t, vel[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_dy.step(t, vel[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_dz.step(t, vel[:, 2], color="C{0}".format(
            n_roll), label="vel [x, y, z] -- trail: {0}".format(n_roll))
        
        ax_euler_x.step(t, euler[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_euler_y.step(t, euler[:, 1], color="C{0}".format(
            n_roll), label="trail :{0}".format(n_roll))
        ax_euler_z.step(t, euler[:, 2], color="C{0}".format(
            n_roll), label="rpy [x, y, z] -- trail: {0}".format(n_roll))
        
        ax_euler_vx.step(t, ang_vel[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_euler_vy.step(t, ang_vel[:, 1], color="C{0}".format(
            n_roll), label="trail :{0}".format(n_roll))
        ax_euler_vz.step(t, ang_vel[:, 2], color="C{0}".format(
            n_roll), label="rpy rates [x, y, z] -- trail: {0}".format(n_roll))

        
        ax_action0.step(t, actions[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_action1.step(t, actions[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_action2.step(t, actions[:, 2], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_action3.step(t, actions[:, 3], color="C{0}".format(
            n_roll), label="act [0, 1, 2, 3] -- trail: {0}".format(n_roll))
        
    ax_z.legend()
    ax_dz.legend()
    ax_euler_z.legend()
    ax_euler_vz.legend()
    ax_action3.legend()
    plt.tight_layout()

    
    #values = np.array([])

    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    z_values = np.asarray(z_values)


    gates = np.array([[-2.0, -2.0, 1.0, -1.0], [-3.0, 3.0, 1.0, 1.0], [3.0, 3.0, 1.0, -2.0], [2.0, -2.0, 2.0, -1.0], [-2.0, -3.0, 2.0, -1.0], [-2.0, -3.0, 1.0, 1.0], [0.0, 0.0, 1.0, -1.0]])
    ax3.plot3D(x_values[500:], y_values[500:], z_values[500:], 'green')
    #ax3.scatter3D(gates[:,0], gates[:,1], gates[:,2], 'red')
    


    gate_width = 0.6
    gate_height = 0.6

    for gate in gates:
        center_x, center_y, center_z, direction = gate
        
        # Determine the vertices of the rectangle based on direction
        if abs(direction) == 2:  # Gate aligned along x-axis
            x = [center_x - gate_width / 2, center_x + gate_width / 2,
                center_x + gate_width / 2, center_x - gate_width / 2]
            y = [center_y, center_y, center_y, center_y]  # Keep y constant
            z = [center_z - gate_height / 2, center_z - gate_height / 2,
                center_z + gate_height / 2, center_z + gate_height / 2]
        else:  # Gate aligned along y-axis
            x = [center_x, center_x, center_x, center_x]  # Keep x constant
            y = [center_y - gate_width / 2, center_y + gate_width / 2,
                center_y + gate_width / 2, center_y - gate_width / 2]
            z = [center_z - gate_height / 2, center_z - gate_height / 2,
                center_z + gate_height / 2, center_z + gate_height / 2]
        
        # Create a polygon for the rectangle
        vertices = [list(zip(x, y, z))]
        gate_poly = Poly3DCollection(vertices, edgecolor='red', facecolor='red', alpha=0.4)
        ax3.add_collection3d(gate_poly)
        #ax3.scatter3D(center_x, center_y, center_z, color='red')


    ax3.set_title('Path of first drone.')


    x_limits = ax3.get_xlim3d()
    y_limits = ax3.get_ylim3d()
    z_limits = ax3.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = np.max([x_range, y_range, z_range])

    # Setting the limits for x, y, z to have the same range
    ax3.set_xlim3d([x_limits[0], x_limits[0] + max_range])
    ax3.set_ylim3d([y_limits[0], y_limits[0] + max_range])
    ax3.set_zlim3d([z_limits[0], z_limits[0] + max_range])

    # Set labels
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    vel = np.empty(len(pos))
    for _ in pos:
        vel[i] = np.linalg.norm(_)


    # Create points for the segments
    """points = np.array([x_values[500:], y_values[500:], z_values[500:]]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a Line3DCollection
    lc = Line3DCollection(segments, cmap='viridis', norm=plt.Normalize(vel.min(), vel.max()))
    lc.set_array(vel)
    lc.set_linewidth(2)
    ax3.add_collection(lc)"""

    plt.show()
    """
    waypoints = np.array([[2, 0], [3, 1], [2, 2], [-1, 3], [-3, 1], [-1, 0], [2, 0], [3, 1], [2, 2], [-1, 3], [-3, 1], [-1, 0], [2, 0], [3, 1], [2, 2], [-1, 3], [-3, 1], [-1, 0], [2, 0], [3, 1], [2, 2], [-1, 3], [-3, 1], [-1, 0],])

    # Number of waypoints
    num_waypoints = len(waypoints)

    # Initial guess for the times at each waypoint
    initial_times = np.linspace(0, 10, num_waypoints)

    # Define the constraints
    max_speed = 3.1  # units per second
    max_acceleration = 4.1  # units per second^2
    max_jerk = 100.0  # units per second^3

    # Function to calculate the spline and its derivatives
    def get_splines_and_derivatives(times):
        tck_x = CubicSpline(times, waypoints[:, 0])
        tck_y = CubicSpline(times, waypoints[:, 1])
        return tck_x, tck_y

    # Define the objective function (total time)
    def objective(times):
        return times[-1]  # Minimize the final time

    # Define the constraints function
    def constraints(times):
        cons = []
        
        # Ensure the times are strictly increasing
        epsilon = 1e-3
        for i in range(1, len(times)):
            cons.append(times[i] - times[i - 1] - epsilon)
        
        tck_x, tck_y = get_splines_and_derivatives(times)
        
        # Speed constraints
        for t in np.linspace(times[0], times[-1], num=len(times)*10):
            dx_dt = tck_x(t, 1)
            dy_dt = tck_y(t, 1)
            speed = np.sqrt(dx_dt**2 + dy_dt**2)
            cons.append(max_speed - speed)
        
        # Acceleration constraints
        for t in np.linspace(times[0], times[-1], num=len(times)*10):
            dx_dt = tck_x(t, 1)
            dy_dt = tck_y(t, 1)
            d2x_dt2 = tck_x(t, 2)
            d2y_dt2 = tck_y(t, 2)
            acceleration = np.sqrt(d2x_dt2**2 + d2y_dt2**2)
            cons.append(max_acceleration - acceleration)
        
        # Jerk constraints
        for t in np.linspace(times[0], times[-1], num=len(times)*10):
            d2x_dt2 = tck_x(t, 2)
            d2y_dt2 = tck_y(t, 2)
            d3x_dt3 = tck_x(t, 3)
            d3y_dt3 = tck_y(t, 3)
            jerk = np.sqrt(d3x_dt3**2 + d3y_dt3**2)
            cons.append(max_jerk - jerk)
        
        return np.array(cons)

    # Set up the constraints dictionary for the optimizer
    constraints_dict = {
        'type': 'ineq',
        'fun': constraints
    }

    # Set up bounds for the times (ensuring they are positive and increasing)
    bounds = [(0, None) for _ in range(num_waypoints)]

    # Run the optimizer
    result = minimize(
        objective,
        initial_times,
        constraints=constraints_dict,
        bounds=bounds,
        method='SLSQP',
        options={'disp': True}
    )

    # Optimized times
    optimal_times = result.x

    # Get the optimized splines
    tck_x_opt, tck_y_opt = get_splines_and_derivatives(optimal_times)

    # Generate the optimized path
    t_new = np.linspace(optimal_times[0], optimal_times[-1], num=num_waypoints*100)
    x_opt = tck_x_opt(t_new)
    y_opt = tck_y_opt(t_new)
    z_opt = np.asarray(np.ones(len(x_opt)))


   
    x_values = np.asarray(x_values[:])#np.asarray(x_values[250:])
    y_values = np.asarray(y_values[:])#np.asarray(y_values[250:])
    z_values = np.asarray(np.ones(len(x_values)))#np.asarray(z_values[250:])#np.asarray(np.ones(len(x_values)))


    #ax2.scatter3D(x_values, y_values, z_values, color='blue', label='Path of first drone')
    
    # Plot orientation using quivers
    for i in range(0, len(x_values), 2):
        x, y, z = x_values[i], y_values[i], z_values[i]
        roll, pitch, yaw = ex_values[i], ey_values[i], ez_values[i]
        # Convert Euler angles to rotation matrix
        R = np.array([
            [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)],
            [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)],
            [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]
        ])
        # Direction vector (orientation) in local frame
        direction = np.dot(R, np.array([1, 0, 0]))
        # Plot quiver
        #ax2.quiver(x, y, z, direction[0], direction[1], direction[2], length=0.2, normalize=True, color='gray', alpha=0.5)

    #ax2.set_title('Path of first drone')
    #ax2.legend()
    #ax2.set_xlabel('X')
    #ax2.set_ylabel('Y')
    #ax2.set_zlabel('Z')

    t_values = np.linspace(np.min(t), np.max(t + 1)/50, 2000)#np.linspace(250/50 + np.min(t), np.max(t + 1)/50, 750)

    #print(t_values)

    print(len(t_values), len(x_values))
    #print(x_values)
    

    spline_x = CubicSpline(t_values, x_values)
    spline_y = CubicSpline(t_values, y_values)
    spline_z = CubicSpline(t_values, z_values)

    x_new = spline_x(t_values)
    y_new = spline_y(t_values)
    z_new = spline_z(t_values)

    #print(x_new)

    ax3.plot3D(x_new, y_new, z_new, color='blue', label='Spline')

    t_values = np.linspace(0, 6.215, 250)


    traj = uav_trajectory.Trajectory()
    traj.loadcsv("traj5.csv")

    evals = np.empty((len(t_values), 15))

    for t, i in zip(t_values, range(0, len(t_values))):
        e = traj.eval(t)
        evals[i, 0:3]  = e.pos
        evals[i, 3:6]  = e.vel
        evals[i, 6:9]  = e.acc
        evals[i, 9:12] = e.omega
        evals[i, 12]   = e.yaw
        evals[i, 13]   = e.roll
        evals[i, 14]   = e.pitch

    
    #ax3.plot3D(evals[:,0], evals[:,1], evals[:,2], color='red', label='optm')
    ax3.plot3D(x_opt, y_opt, z_opt, color='red', label='optm')

    """
    #plt.show()


