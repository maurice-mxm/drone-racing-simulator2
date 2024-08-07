#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <random>
#include <cmath>
#include <array>
#include <numeric>
#include <tuple>


namespace py = pybind11;

class BaseEnvironment {
public:
    BaseEnvironment() : state(12), old_control(4) {
        std::random_device rd;
        std::mt19937 gen(rd());

        // Initialize state
        std::uniform_real_distribution<> dist1(-3.0, 3.0);
        std::uniform_real_distribution<> dist2(0.0, 3.0);
        std::uniform_real_distribution<> dist3(1.2, 1.4);
        std::uniform_real_distribution<> dist4(-0.001, 0.001);

        state[0] = dist1(gen);
        state[1] = dist2(gen);
        state[2] = dist3(gen);

        for (size_t i = 3; i < state.size(); ++i) {
            state[i] = dist4(gen);
        }

        initial_quat = Dynamics::euler_to_quaternion(state[6], state[7], state[8]);
        quaternion = initial_quat;

        std::tuple<std::array<double, 12>, double, bool> dynamics(
        const std::array<double, 12>& state,
        const std::array<double, 4>& control,
        const std::array<double, 4>& old_control
        
    );

    }

    int get_observation_dimension() const {
        return 12;
    }

    int get_action_dimension() const {
        return 4;
    }

    std::vector<double> reset() {
        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_real_distribution<> dist1(-3.0, 3.0);
        std::uniform_real_distribution<> dist2(0.0, 3.0);
        std::uniform_real_distribution<> dist3(1.2, 1.4);
        std::uniform_real_distribution<> dist4(-0.001, 0.001);

        state[0] = dist1(gen);
        state[1] = dist2(gen);
        state[2] = dist3(gen);

        for (size_t i = 3; i < state.size(); ++i) {
            state[i] = dist4(gen);
        }

        quaternion = initial_quat;

        return state;
    }

    std::tuple<std::vector<double>, double, bool> step(const std::vector<double>& action) {
        auto [new_state, reward, done] = drone_dynamics2::dynamics(state, action, old_control);
        old_control = action;
        state = new_state;

        return std::make_tuple(state, reward, done);
    }

    std::vector<double> get_observation() const {
        return state;
    }

private:
    std::vector<double> state;
    std::vector<double> old_control;
    std::vector<double> initial_quat;
    std::vector<double> quaternion;
};

class VectorEnvironment {
public:
    VectorEnvironment(const std::map<std::string, int>& config) {
        number_of_envs = config.at("nenvs");
        envs.resize(number_of_envs);

        obs_dim = envs[0].get_observation_dimension();
        act_dim = envs[0].get_action_dimension();
    }

    int getObsDim() const {
        return obs_dim;
    }

    int getActDim() const {
        return act_dim;
    }

    int getNumEnvs() const {
        return number_of_envs;
    }

    std::vector<std::vector<double>> reset(int seed = 0) {
        std::vector<std::vector<double>> observations(number_of_envs);
        for (int i = 0; i < number_of_envs; ++i) {
            observations[i] = envs[i].reset();
        }
        return observations;
    }

    std::tuple<std::vector<std::vector<double>>, std::vector<double>, std::vector<bool>>
    step(const std::vector<std::vector<double>>& actions) {
        std::vector<std::vector<double>> observation(number_of_envs);
        std::vector<double> reward(number_of_envs);
        std::vector<bool> done(number_of_envs);

        for (int i = 0; i < number_of_envs; ++i) {
            auto [obs, rew, d] = envs[i].step(actions[i]);
            observation[i] = obs;
            reward[i] = rew;
            done[i] = d;
        }

        return std::make_tuple(observation, reward, done);
    }

private:
    int number_of_envs;
    int obs_dim;
    int act_dim;
    std::vector<BaseEnvironment> envs;
};


class Dynamics {
public:
    // Constructor to initialize the environment
    Dynamics();

    // Public method to compute dynamics
    

private:
    // Constants
    static constexpr double GRAVITY = 9.81;
    static constexpr double THRUST_TO_WEIGHT = 2.25;
    static constexpr double TIMESTEP = 0.005;
    static constexpr double MASS = 0.028;
    static constexpr double KF = 3.16e-10;
    static constexpr double KM = 7.94e-12;
    static constexpr double L = 0.0397;
    static constexpr double HOVER_RPM = 16000;
    static constexpr double MAX_RPM = 24000;
    static constexpr double AIR_DENSITY = 1.225e03;
    static constexpr double DRAG_COEFFICIENT = 0.0000806428;
    static constexpr double REFERENCE_AREA = 0.1;

    static const std::array<std::array<double, 3>, 3> J;

    // Helper methods
    std::array<double, 4> quaternionFromEuler(double roll, double pitch, double yaw);
    std::array<double, 4> quaternionMultiply(const std::array<double, 4>& q1, const std::array<double, 4>& q2);
    std::array<double, 3> quaternionRotate(const std::array<double, 4>& q, const std::array<double, 3>& v);
    double inputToThrust(double input);
    double forceToTorque(double force);
    double inputToRotVel(double input);
};

// Initialize static constant J
const std::array<std::array<double, 3>, 3> VectorEnvironment::J = {{
    {16.57e-6, 0.83e-6, 0.718e-6},
    {0.83e-6, 16.655e-6, 1.8e-6},
    {0.718e-6, 1.8e-6, 29.26e-6}
}};

    // Constructor
    VectorEnvironment::VectorEnvironment() {
        // Any initialization can be done here
    }

    // Helper method implementations
    std::array<double, 4> VectorEnvironment::quaternionFromEuler(double roll, double pitch, double yaw) {
        double cr = std::cos(roll / 2);
        double sr = std::sin(roll / 2);
        double cp = std::cos(pitch / 2);
        double sp = std::sin(pitch / 2);
        double cy = std::cos(yaw / 2);
        double sy = std::sin(yaw / 2);

        return {cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy};
    }

    std::array<double, 4> VectorEnvironment::quaternionMultiply(const std::array<double, 4>& q1, const std::array<double, 4>& q2) {
        double w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
        double w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];

        return {w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2};
    }

    std::array<double, 3> VectorEnvironment::quaternionRotate(const std::array<double, 4>& q, const std::array<double, 3>& v) {
        std::array<double, 4> q_conj = {q[0], -q[1], -q[2], -q[3]};
        std::array<double, 4> v_as_quat = {0, v[0], v[1], v[2]};
        auto rotated_v = quaternionMultiply(quaternionMultiply(q, v_as_quat), q_conj);
        return {rotated_v[1], rotated_v[2], rotated_v[3]};
    }

    double VectorEnvironment::inputToThrust(double input) {
        return 2.130295e-11 * input * input + 1.032633e-06 * input + 5.484560e-04;
    }

    double VectorEnvironment::forceToTorque(double force) {
        return 5.964552e-03 * force + 1.563383e-05;
    }

    double VectorEnvironment::inputToRotVel(double input) {
        return 4.076521e-02 * input + 380.8359;
    }

    // Dynamics function
    std::tuple<std::array<double, 12>, double, bool> VectorEnvironment::dynamics(
        const std::array<double, 12>& state,
        const std::array<double, 4>& control,
        const std::array<double, 4>& old_control) {

        // Unpack the state vector
        std::array<double, 3> position = {state[0], state[1], state[2]};
        std::array<double, 3> velocity = {state[3], state[4], state[5]};
        std::array<double, 3> orientation = {state[6], state[7], state[8]};
        std::array<double, 3> angular_velocity = {state[9], state[10], state[11]};
        auto quaternion = quaternionFromEuler(orientation[0], orientation[1], orientation[2]);

        // Convert control signals to input values
        std::array<double, 4> inputs;
        for (size_t i = 0; i < 4; ++i) {
            inputs[i] = (control[i] * 32750 + 32750);
        }

        // Calculate forces generated by the motors
        std::array<double, 4> forces = {inputToThrust(inputs[0]), inputToThrust(inputs[1]), inputToThrust(inputs[2]), inputToThrust(inputs[3])};
        std::array<double, 3> force = {0, 0, std::accumulate(forces.begin(), forces.end(), 0.0)};

        // Calculate torques generated by the motors
        std::array<double, 4> torques = {forceToTorque(forces[0]), forceToTorque(forces[1]), forceToTorque(forces[2]), forceToTorque(forces[3])};

        // Calculate rotational velocities
        std::array<double, 4> rot_vel = {inputToRotVel(inputs[0]), inputToRotVel(inputs[1]), inputToRotVel(inputs[2]), inputToRotVel(inputs[3])};

        // Calculate torque components in x, y, z directions
        double torque_x = (-torques[0] - torques[1] + torques[2] + torques[3]) * (L / std::sqrt(2));
        double torque_y = (-torques[0] + torques[1] + torques[2] - torques[3]) * (L / std::sqrt(2));
        double torque_z = (-torques[0] + torques[1] - torques[2] + torques[3]);
        std::array<double, 3> torque = {torque_x, torque_y, torque_z};

        // Calculate linear and angular accelerations
        auto acc = quaternionRotate(quaternion, force) / MASS + std::array<double, 3>{0, 0, -GRAVITY};
        std::array<double, 3> angular_acc = {0, 0, 0}; // To be calculated if necessary

        // Update state variables
        for (size_t i = 0; i < 3; ++i) {
            velocity[i] += acc[i] * TIMESTEP;
            position[i] += velocity[i] * TIMESTEP;
            angular_velocity[i] += angular_acc[i] * TIMESTEP;
        }

        // Update orientation (assuming small angles for simplicity)
        for (size_t i = 0; i < 3; ++i) {
            orientation[i] += angular_velocity[i] * TIMESTEP;
        }

        // Pack the new state
        std::array<double, 12> new_state = {position[0], position[1], position[2],
                                            velocity[0], velocity[1], velocity[2],
                                            orientation[0], orientation[1], orientation[2],
                                            angular_velocity[0], angular_velocity[1], angular_velocity[2]};

        // Placeholder for reward calculation
        double total_reward = 0.0; // Calculate reward based on new state and task-specific objectives
        bool done = false;         // Determine if the episode is done based on criteria

        return std::make_tuple(new_state, total_reward, done);
}


PYBIND11_MODULE(vec_processor, m) {
    py::class_<BaseEnvironment>(m, "BaseEnvironment")
        .def(py::init<>())
        .def("get_observation_dimension", &BaseEnvironment::get_observation_dimension)
        .def("get_action_dimension", &BaseEnvironment::get_action_dimension)
        .def("reset", &BaseEnvironment::reset)
        .def("step", &BaseEnvironment::step)
        .def("get_observation", &BaseEnvironment::get_observation);

    py::class_<VectorEnvironment>(m, "VectorEnvironment")
        .def(py::init<const std::map<std::string, int>&>())
        .def("getObsDim", &VectorEnvironment::getObsDim)
        .def("getActDim", &VectorEnvironment::getActDim)
        .def("getNumEnvs", &VectorEnvironment::getNumEnvs)
        .def("reset", &VectorEnvironment::reset)
        .def("step", &VectorEnvironment::step);
}
