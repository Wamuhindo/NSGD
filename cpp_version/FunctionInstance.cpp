//
// Created by Xueyuan Huang on 20/10/25.
//
// FunctionInstance.cpp
#include "SimProcess.h"
#include "FunctionInstance.h"
#include <stdexcept>
#include <limits>

FunctionInstance::FunctionInstance(double creation_t,
                                   const SimProcess& coldStart,
                                   const SimProcess& coldService,
                                   const SimProcess& warmService,
                                   const SimProcess& expiration)
        : m_coldStart(&coldStart),
          m_coldService(&coldService),
          m_warmService(&warmService),
          m_expiration(&expiration),
          m_creation_time(creation_t) {

          m_state = FunctionState::COLD;
          m_next_init_departure = std::numeric_limits<double>::infinity();
          m_next_departure = std::numeric_limits<double>::infinity();
          m_next_termination = std::numeric_limits<double>::infinity();

          generate_init_departure(creation_t);
          update_next_termination();
}

//cold 到 init free 安排init 完成时间
void FunctionInstance::make_Init_Free(double t){
    if (m_state != FunctionState::COLD)
        throw std::runtime_error("make_Init_Free() requires state COLD");
    m_state = FunctionState::INIT_FREE;
    generate_init_departure(t);
}

// cold到 init reserved 安排init 完成时间并标记为已预订
void FunctionInstance::make_Init_Reserved(double t) {
    if (m_state != FunctionState::COLD)
        throw std::runtime_error("make_init_Reserved() requires state COLD");
    m_state = FunctionState::INIT_RESERVED;
    m_reserved = true;
    generate_init_departure(t);
}

//仅在init free 可预订
bool FunctionInstance::reserve() {
    if (m_state != FunctionState::INIT_FREE) return false;
    if (m_reserved) return false;
    m_reserved = true;
    return true;
}

// 安排init完成时间，若已预订，联动安排第一单的departure，否则先不安排
void FunctionInstance::generate_init_departure(double t) {
    m_next_init_departure = t + m_coldStart->sample();
    if (m_reserved) {
        m_next_departure = m_next_init_departure + m_warmService->sample();
    }
    else {
        m_next_departure = m_next_init_departure;
    }
}

//进入idle on后 根据当前=已完成时刻 安排过期 **
void FunctionInstance::update_next_termination() {
    m_next_termination = m_next_departure + m_expiration->sample();
}

// 当init free已预订且init完成后 衔接第一单
void FunctionInstance::update_next_transition(double t) {
    m_next_departure = m_next_init_departure + m_warmService->sample();
}

//根据当前状态决定是否接单
bool FunctionInstance::arrival_transition(double t) {
    if (is_cold() || is_busy()) return false;
    else if (is_idle_on()) {
        m_state = FunctionState::BUSY;
        m_next_departure = t + m_warmService->sample();
        update_next_termination();
        return true;
    }
    else if (is_init_free() && !m_reserved) {
        return reserve();
    }
    return false; //其他状态 不接受新请求
}

FunctionState FunctionInstance::make_transition() {

    if (is_busy()) {
        m_state = FunctionState::IDLE_ON;
        update_next_termination(); //计算过期时间
    }
    else if (is_init_reserved() || (is_init_free() && m_reserved)) {
        m_state = FunctionState ::BUSY;
    }
    else if (is_init_free() && !m_reserved) {
        m_state = FunctionState::IDLE_ON;
        update_next_termination();
    }
    else if (is_idle_on()) {
        m_state = FunctionState::COLD;
    }
    else {
        throw std::runtime_error("make_transition(): invalid state for transition");
    }
    return m_state;
}

double FunctionInstance::get_next_transition_time(double t) const {
    constexpr double eps = 1e-12;

    if (m_state == FunctionState::IDLE_ON){
        if ( t > m_next_termination + eps ) throw std::runtime_error("after termination");
        return std::max(0.0, m_next_termination - t);
    }
    if (m_state == FunctionState::INIT_FREE || m_state == FunctionState::INIT_RESERVED){
        if (t > m_next_init_departure + eps) throw std::runtime_error("after departure");
        return std::max(0.0, m_next_init_departure - t);
    }
    if (m_state == FunctionState::BUSY){
        if (t > m_next_departure + eps) throw std::runtime_error("after busy departure");
        return std::max(0.0, m_next_departure - t);
    }
    if (m_state == FunctionState::COLD){
        return std::numeric_limits<double>::infinity();
    }
    throw std::runtime_error("unknown state in get_next_transition_time");
}