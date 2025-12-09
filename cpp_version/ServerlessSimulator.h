//
// Created by Xueyuan Huang on 16/10/25.
//

#ifndef UNTITLED2_SERVERLESSSIMULATOR_H
#define UNTITLED2_SERVERLESSSIMULATOR_H
#include <vector>
#include <array>

#include "FunctionInstance.h"
#include "SimProcess.h"
#include "AutoScalingAlgorithm.h"

class ServerlessSimulator {
public:
    struct Stats {
        long total_req{0}, total_cold{0}, total_warm{0}, total_reject{0};
        long total_init_free{0}, total_init_reserved{0}, total_queued{0};

        double sum_lifespan{0.0};
        long num_terminated{0};
    };
    struct Hist {
        std::vector<double> times;
        std::vector<int> server_count, running_count, idle_count, init_free_count, init_reserved_count, queued_job_count;
        std::vector<int> req_cold_idx, req_warm_idx, req_rej_idx, req_init_free_idx, req_init_reserved_idx, req_queued_idx;
    };

    //构造：注入四类过程和到达过程，容量与autoscaler
    ServerlessSimulator(ExpSimProcess arrival,
                        SimProcess& cold_start,
                        SimProcess& cold_service,
                        SimProcess& warm_service,
                        SimProcess& expiration,
                        int max_concurrency,
                        AutoScalingAlgorithm algo);

    //初始化
    void initialize_system(double t0, int running_cnt, int idle_cnt, int init_free_cnt, int init_reserved_cnt);

    //主仿真
    void run(bool progress=false, bool debug=false);

    //统计
    const Stats& stats() const{ return stats_; }
    const Hist& hist() const{ return hist_;}

    //打印结果
    void print_sunmmary() const;

private:
    //调度与操作
    bool has_server() const { return !servers_.empty();}
    bool is_warm_available() const { return idle_count_ > 0;}
    bool is_init_free_available() const {return init_free_count_ > 0;}

    //到达分派
    void warm_start_arrival(double t, int theta);
    void init_free_arrival(double t, int theta);
    void cold_start_arrival(double t, int theta);

    //库存维护 在到达后触发
    void start_init_free_servers(double t, int theta, const char* why);

    //选择器
    int pick_newest_idle() const;
    int pick_newest_init_free() const;

    //状态更新，日志
    void update_hist_arrays(double t);
    void update_state_vector();

private:
    ExpSimProcess arrival_;
    SimProcess& cold_start_;
    SimProcess& cold_service_;
    SimProcess& warm_service_;
    SimProcess& expiration_;

    //autoscaler 简化
    AutoScalingAlgorithm algo_;

    //容量与实例池
    int max_conc_{0};
    std::vector<FunctionInstance> servers_;

    //计数
    int server_count_{0}, running_count_{0}, idle_count_{0};
    int init_free_count_{0}, init_reserved_count_{0}, init_free_booked_{0}, queued_jobs_count_{0};

    //历史与统计
    Hist hist_;
    Stats stats_;

    //系统状态 传给autoscaler
    std::array<int, 5> state_{};

    //时间
    double t_{0.0};
    double next_arrival_{0.0};
};

#endif //UNTITLED2_SERVERLESSSIMULATOR_H
