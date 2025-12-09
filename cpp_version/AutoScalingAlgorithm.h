//
// Created by Xueyuan Huang on 22/10/25.
//

#ifndef UNTITLED2_AUTOSCALINGALGORITHM_H
#define UNTITLED2_AUTOSCALINGALGORITHM_H

#include <array>
#include <vector>

enum class SystemStateCpp {
    COLD = 0,
    IDLE_ON = 1,
    INITIALIZING = 2,
    BUSY = 3,
    INIT_RESERVED = 4
};

class AutoScalingAlgorithm {
public:
    AutoScalingAlgorithm(
            int N,
            double k_delta,
            const std::array<double,3>& k_gamma,
            const std::array<double,3>& theta_init,
            double tau,
            int K,
            double T_max
            );
    //python running condition
    bool running_condition() const;

    void advance_to(double t);

    void set_has_rejected_job(bool v);

    std::array<int,2> get_theta_step() const;

    double compute_cost(const std::array<int,5>& state);

    template <typename Simulator>
    void simulate_step(const std::array<int,5>& state, Simulator& sim)
    {
        (void)sim; //未使用的参数
        (void)state;
        t_ += 1.0; //暂时只当成 时间步进
    }
private:
    void init_state();

    void init_weights();

private:
    int N_;
    double k_delta_;
    std::array<double,3> k_gamma_;
    double tau_;
    int K_;
    double Tmax_;

    std::array<double,3> opt_; //当前最佳theta
    std::array<double,3> opt_init_; //初始theta
    std::array<double,3> opt_step_; //theta step

    std::array<int,5> state_; //当前系统状态向量
    std::array<double,5> weights_; //cost权重
    double w_rej_; //拒绝请求惩罚项

    bool has_rejected_job_;

    //时间与迭代计数
    double t_;
    int n_;
    int k_;

    double last_cost_;
};


#endif //UNTITLED2_AUTOSCALINGALGORITHM_H
