//
// Created by Xueyuan Huang on 05/12/25.
//

#include "AutoScalingAlgorithm.h"
#include <algorithm>
#include <cmath>

AutoScalingAlgorithm::AutoScalingAlgorithm(int N, double k_delta, const std::array<double, 3> &k_gamma,
                                           const std::array<double, 3> &theta_init, double tau, int K, double T_max)
                                           :
                                           N_(N),
                                           k_delta_(k_delta),
                                           k_gamma_(k_gamma),
                                           tau_(tau),
                                           K_(K),
                                           Tmax_(T_max),
                                           opt_(theta_init),
                                           opt_init_(theta_init),
                                           opt_step_(theta_init),
                                           has_rejected_job_(false),
                                           t_(0.0),
                                           n_(1),
                                           k_(0),
                                           last_cost_(0.0)
                                           { init_state(); init_weights();}

bool AutoScalingAlgorithm::running_condition() const {
    return t_ < Tmax_;
}

void AutoScalingAlgorithm::advance_to(double t) {
    t_ = t;
}

void AutoScalingAlgorithm::set_has_rejected_job(bool v) {
    has_rejected_job_ = v;
}

std::array<int,2> AutoScalingAlgorithm::get_theta_step() const {

        int theta = static_cast<int>(opt_step_[0]);
        int theta_min = static_cast<int>(opt_step_[1]);
        if (theta < 0) theta = 0;
        if (theta_min < 0) theta_min = 0;
        if (theta > N_) theta = N_;
        if (theta_min > N_) theta_min = N_;
        return { theta, theta_min};

}

double AutoScalingAlgorithm::compute_cost(const std::array<int, 5> &state) {
    state_ = state;
    double cost = 0.0;
    for (int i = 0; i < 5; ++i){
        cost += state_[i] * weights_[i];
    }
    if (has_rejected_job_) {
        cost += w_rej_;
    }
    last_cost_ = cost;
    return cost;
}

void AutoScalingAlgorithm::init_state() {
    state_.fill(0);
    state_[static_cast<int>(SystemStateCpp::COLD)] = N_;
}

void AutoScalingAlgorithm::init_weights() {
    weights_.fill(0.0);

    weights_[static_cast<int>(SystemStateCpp::COLD)] = 0.0;
    weights_[static_cast<int>(SystemStateCpp::IDLE_ON)] = 1.0;
    weights_[static_cast<int>(SystemStateCpp::INITIALIZING)] = 5.0;
    weights_[static_cast<int>(SystemStateCpp::BUSY)] = 1.0;
    weights_[static_cast<int>(SystemStateCpp::INIT_RESERVED)] = 50.0;

    w_rej_ = 100.0;
}




