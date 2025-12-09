//
// Created by Xueyuan Huang on 20/10/25.
//

#include "SimProcess.h"
#include "FunctionInstance.h"
#include "ServerlessSimulator.h"

#include <algorithm>
#include <limits>
#include <iomanip>

ServerlessSimulator::ServerlessSimulator (ExpSimProcess arrival,
                                          SimProcess& cold_start,
                                          SimProcess& cold_service,
                                          SimProcess& warm_service,
                                          SimProcess& expiration,
                                         int max_concurrency,
                                         AutoScalingAlgorithm algo):
                                         arrival_(std::move(arrival)),
                                         cold_start_(cold_start),
                                         cold_service_(cold_service),
                                         warm_service_(warm_service),
                                         expiration_(expiration),
                                         algo_(std::move(algo)),
                                         max_conc_(max_concurrency){
    //首次到达
    next_arrival_ = t_ + arrival_.sample();
}

void ServerlessSimulator::initialize_system(double t0, int running_cnt, int idle_cnt, int init_free_cnt, int init_reserved_cnt) {
    t_ = t0;
    servers_.reserve(running_cnt + idle_cnt + init_free_cnt + init_reserved_cnt);

    //构造idle 实例
    for (int i=0; i<idle_cnt; ++i){
        FunctionInstance f(t_, cold_start_, cold_service_, warm_service_, expiration_);
        f.make_Init_Free(t0);
        f.make_transition();
        servers_.push_back(std::move(f));
        ++idle_count_;
        ++server_count_;
    }

    //构造running实例
    for (int i=0; i<running_cnt; ++i){
        FunctionInstance f(t_, cold_start_, cold_service_, warm_service_, expiration_);
        f.make_Init_Free(t0);
        f.make_transition();
        f.arrival_transition(t_);
        servers_.push_back(std::move(f));
        ++running_count_;
        ++server_count_;
    }

    //构造init free实例
    for (int i=0; i<init_free_cnt; ++i){
        FunctionInstance f(t_, cold_start_, cold_service_, warm_service_, expiration_);
        f.make_Init_Free(t0);
        servers_.push_back(std::move(f));
        ++init_free_count_;
        ++server_count_;
    }

    //构造init reserved实例
    for (int i=0; i<init_reserved_cnt; ++i){
        FunctionInstance f(t_, cold_start_, cold_service_, warm_service_, expiration_);
        f.make_Init_Reserved(t0);
        servers_.push_back(std::move(f));
        ++init_reserved_count_;
        ++server_count_;
    }
}

int ServerlessSimulator::pick_newest_idle() const {
    int idx = -1;
    double best_ct = -1e300;
    for (int i=(int)servers_.size()-1;i>=0; --i){
        if (servers_[i].is_idle_on()) return i;
    }
    return idx;
}

int ServerlessSimulator::pick_newest_init_free() const {

    for (int i=(int)servers_.size()-1;i>=0; --i){
        const auto& s = servers_[i];
        if (s.is_init_free() && !s.is_reserved()) return i;
    }
    return -1;
}

void ServerlessSimulator::update_hist_arrays(double t) {
    hist_.times.push_back(t);
    hist_.server_count.push_back(server_count_);
    hist_.running_count.push_back(running_count_);
    hist_.idle_count.push_back(idle_count_);
    hist_.init_reserved_count.push_back(init_reserved_count_);
    hist_.init_free_count.push_back(init_free_count_);
    hist_.queued_job_count.push_back(queued_jobs_count_);
}

void ServerlessSimulator::update_state_vector() {
    state_.fill(0);
    state_[0] = init_free_count_ + init_reserved_count_ + init_free_booked_;
    state_[1] = init_reserved_count_ + init_free_booked_;
    state_[2] = running_count_;
    state_[3] = idle_count_;
    state_[4] = max_conc_ - (init_free_count_ + init_free_booked_ + init_reserved_count_ + running_count_ + idle_count_);
}

void ServerlessSimulator::start_init_free_servers(double t, int theta, const char *why) {
    if (theta < 0) return;
    int current_conc = running_count_ + init_reserved_count_ + init_free_booked_;
    int cold_servers = max_conc_ - (init_free_count_ + init_free_booked_ + init_reserved_count_ + running_count_ + idle_count_);
    int pi_theta = std::max(0, theta - init_free_count_);
    int to_start = (current_conc + pi_theta > max_conc_) ? (max_conc_ - current_conc) : pi_theta;
    to_start = std::min(to_start, cold_servers);

    for (int i=0; i<to_start; ++i){
        FunctionInstance f(t, cold_start_, cold_service_, warm_service_, expiration_);
        f.make_Init_Free(t);
        servers_.push_back(std::move(f));
        ++init_free_count_;
        ++server_count_;
        ++stats_.total_init_free;
        if (!hist_.times.empty()) hist_.req_init_free_idx.push_back((int)hist_.times.size()-1);
    }
}

void ServerlessSimulator::warm_start_arrival(double t, int theta) {
    ++stats_.total_req;
    int current_conc = running_count_ + init_reserved_count_ + init_free_booked_;
    if (current_conc >= max_conc_){
        ++stats_.total_reject;
        if (!hist_.times.empty()) hist_.req_rej_idx.push_back((int)hist_.times.size()-1);
        algo_.set_has_rejected_job(true);
        return;
    }

    int idx = pick_newest_idle();
    if (idx < 0){
        cold_start_arrival(t, theta);
        return;
    }

    bool was_idle = servers_[idx].is_idle_on();
    servers_[idx].arrival_transition(t);
    ++stats_.total_warm;
    if (was_idle){--idle_count_; ++running_count_;}
    if (!hist_.times.empty()) hist_.req_warm_idx.push_back((int)hist_.times.size()-1);

    //维持库存 如果idle低于阈值 补充init free
    auto step = algo_.get_theta_step();
    int theta_idle = step[1];
    if (idle_count_ < theta_idle){
        int to_start = step[0] - idle_count_;
        if (to_start > 0)
            start_init_free_servers(t, to_start, "Warm start");

    }
}

void ServerlessSimulator::init_free_arrival(double t, int theta) {
    ++stats_.total_req;
    ++stats_.total_queued;
    ++queued_jobs_count_;

    int current_conc = running_count_ + init_reserved_count_ + init_free_booked_;
    if (current_conc >= max_conc_){
        ++stats_.total_reject;
        if (!hist_.times.empty()) hist_.req_rej_idx.push_back((int)hist_.times.size()-1);
        algo_.set_has_rejected_job(true);
        return;
    }
    int idx;
    idx = pick_newest_init_free();
    if (idx < 0){
        ++stats_.total_reject;
        if (!hist_.times.empty()) hist_.req_rej_idx.push_back((int)hist_.times.size()-1);
        algo_.set_has_rejected_job(true);
        return;
    }
    servers_[idx].arrival_transition(t);
    --init_free_count_;
    ++init_free_booked_;
    ++stats_.total_cold;
    if (!hist_.times.empty()) hist_.req_queued_idx.push_back((int)hist_.times.size()-1);
    start_init_free_servers(t, theta, "Init free arrival");
}

void ServerlessSimulator::cold_start_arrival(double t, int theta) {
    ++stats_.total_req;
    int current_conc = running_count_ + init_reserved_count_ + init_free_booked_;
    int cold_servers = max_conc_ - (init_free_count_ + init_free_booked_ + init_reserved_count_ + running_count_ + idle_count_);
    if (current_conc >= max_conc_ || cold_servers <= 0){
        ++stats_.total_reject;
        if (!hist_.times.empty()) hist_.req_rej_idx.push_back((int)hist_.times.size()-1);
        algo_.set_has_rejected_job(true);
        return;
    }
    ++stats_.total_cold;
    if (!hist_.times.empty()) {
        hist_.req_cold_idx.push_back((int) hist_.times.size() - 1);
        hist_.req_init_reserved_idx.push_back((int) hist_.times.size() - 1);
    }

    FunctionInstance f(t, cold_start_, cold_service_, warm_service_, expiration_);
    f.make_Init_Reserved(t);
    servers_.push_back(std::move(f));
    ++init_reserved_count_;
    ++server_count_;
    ++stats_.total_init_reserved;

    start_init_free_servers(t_, theta, "Cold start");

}

void ServerlessSimulator::run(bool progress, bool debug) {
    auto step = algo_.get_theta_step();
    start_init_free_servers(t_, step[0], "bootstrap");

    while (algo_.running_condition()){
        //选择下一事件时间：到达 vs 最早实例事件
        double min_inst_dt = std::numeric_limits<double>::infinity();
        int min_idx = -1;
        for (int i=0; i<(int)servers_.size();++i){
            double dt = servers_[i].get_next_transition_time(t_);
            if (dt < min_inst_dt){ min_inst_dt = dt; min_idx = i;}
        }
        double next_event_t = t_ + min_inst_dt;

        // 到达更早？
        if (!has_server() || next_arrival_ < next_event_t) {
            t_ = next_arrival_;
            hist_.times.push_back(t_);
            update_hist_arrays(t_);

            step = algo_.get_theta_step();
            int theta = step[0];

            if (is_warm_available()) {
                warm_start_arrival(t_, theta);
            }
            else if (is_init_free_available()){
                init_free_arrival(t_, theta);
            }
            else{
                cold_start_arrival(t_, theta);
            }
            // 下一次到达
            next_arrival_ = t_ + arrival_.sample();

            //更新autoscaler
            update_state_vector();
            algo_.advance_to(t_);
            (void)algo_.compute_cost(state_);
            algo_.simulate_step(state_,*this);
            continue;
        }
        // 实例事件更早
        t_ = next_event_t;
        hist_.times.push_back(t_);
        update_hist_arrays(t_);

        auto old_state = servers_[min_idx].state();
        auto new_state = servers_[min_idx].make_transition();

        if (new_state == FunctionState::COLD){
            if (old_state == FunctionState::IDLE_ON) {
                --idle_count_;
            }
            //统计寿面：当前时间-创建时间
            stats_.sum_lifespan += (t_ - servers_[min_idx].creation_time());
            stats_.num_terminated += 1;

            --server_count_;
            servers_.erase(servers_.begin()+min_idx);
        }
        else if (new_state == FunctionState::IDLE_ON){
            if (old_state == FunctionState::BUSY){
                --running_count_;
                ++idle_count_;
            }
            else if (old_state == FunctionState::INIT_FREE){
                --init_free_count_;
                ++idle_count_;
                servers_[min_idx].unreserved();
            }
        }
        else if (new_state == FunctionState::BUSY){
            if (old_state == FunctionState::INIT_RESERVED){
                --init_reserved_count_;
                ++running_count_;
            }
            else if (old_state == FunctionState::INIT_FREE && servers_[min_idx].is_reserved()){
                servers_[min_idx].update_next_transition(t_);
                servers_[min_idx].unreserved();

                --init_free_booked_;
                --queued_jobs_count_;
                ++stats_.total_warm;
                hist_.req_warm_idx.push_back((int)hist_.times.size()-1);
                ++running_count_;
            }
        }
        update_state_vector();
        algo_.advance_to(t_);
        (void)algo_.compute_cost(state_);
        algo_.simulate_step(state_, *this);
    }
}

static double avg_of(const std::vector<int>& v){
    if (v.empty()) return 0.0;
    double s = std::accumulate(v.begin(), v.end(), 0.0);
    return s / v.size();
}

void ServerlessSimulator::print_sunmmary() const {
    //基本比例
    const double total_req = static_cast<double>(stats_.total_req);
    const double p_cold = (total_req > 0) ? (double)stats_.total_cold / total_req : 0.0;
    const double p_rej = (total_req > 0) ? (double)stats_.total_reject / total_req : 0.0;

    //历史平均
    double avg_server = avg_of(hist_.server_count);
    double avg_run = avg_of(hist_.running_count);
    double avg_idle = avg_of(hist_.idle_count);
    double avg_ifree = avg_of(hist_.init_free_count);
    double avg_ires = avg_of(hist_.init_reserved_count);
    double avg_qjobs = avg_of(hist_.queued_job_count);
    double avg_life = stats_.num_terminated ? (stats_.sum_lifespan / stats_.num_terminated) : 0.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Cold starts / total requests:" << stats_.total_cold << " / " << stats_.total_req << "\n";
    std::cout << "Cold Start Probability:      " << p_cold << "\n";
    std::cout << "Rejection / total requests:  " << stats_.total_reject << " / " << stats_.total_req << "\n";
    std::cout << "Rejection Probability:       " << p_rej << "\n";

    //这些项目需要在模拟里面做更加精细的记录才能得出，先占位
    std::cout << "Average Instance Life Span: " << avg_life << "\n";

    std::cout << "Average Server Count:       " << avg_server << "\n";
    std::cout << "Average Running Count:      " << avg_run << "\n";
    std::cout << "Average Idle Count:         " << avg_idle << "\n";
    std::cout << "Average Init Free Count:    " << avg_ifree << "\n";
    std::cout << "Average Init Reserved Count:" << avg_ires << "\n";
    std::cout << "Average Queued Jobs Count:  " << avg_qjobs << "\n";

    std::cout << "Total Init Free Count:      " << stats_.total_init_free << "\n";
    std::cout << "Total Init Reserved Count:  " << stats_.total_init_reserved << "\n";
    std::cout << "Total Queued Jobs Count:    " << stats_.total_queued << "\n";
}

