//
// Created by Xueyuan Huang on 15/10/25.
//

#ifndef UNTITLED2_FUNCTIONINSTANCE_H
#define UNTITLED2_FUNCTIONINSTANCE_H

#include <optional>
#include <string>
#include <iostream>
#include <limits>

#include "SimProcess.h"

//统一状态枚举
enum class FunctionState { COLD, INIT_FREE, INIT_RESERVED, IDLE_ON, BUSY};

// 单个函数实例的状态
class FunctionInstance {
public:

    // 四个独立过程：冷启动，首单服务，后续服务，空闲过期
    FunctionInstance(double creation_t,
                     const SimProcess &coldStart,
                     const SimProcess &coldService,
                     const SimProcess &warmService,
                     const SimProcess &expiration);

    //事件驱动接口
    void generate_init_departure(double t); //生成初始化完成的时间点
    void update_next_termination(); //进入idle on以后生成的过期时间
    void update_next_transition(double t); //当init free被预订且init完成后，衔接第一单

    // 到达与转移
    bool arrival_transition(double t); //返回是否接受（并更新时间戳）
    FunctionState make_transition(); // 内部状态推进，在恰好发生事件时调用，返回新状态

    // 工具
    bool reserve(); //仅init free时可reserve，成功返回true
    void unreserved() { m_reserved = false; }

    //查询器
    [[nodiscard]] FunctionState state() const { return m_state; }

    bool is_cold() const { return m_state == FunctionState::COLD; }

    bool is_idle_on() const { return m_state == FunctionState::IDLE_ON; }

    bool is_busy() const { return m_state == FunctionState::BUSY; }

    bool is_init_free() const { return m_state == FunctionState::INIT_FREE; }

    bool is_init_reserved() const { return m_state == FunctionState::INIT_RESERVED; }

    bool is_reserved() const { return m_reserved; }

    double creation_time() const { return m_creation_time; }

    // 设置状态 仅用于构造特殊初始场景
    void make_Init_Free(double t); //cold--init free
    void make_Init_Reserved(double t); //cold--init reserved

    double get_next_transition_time(double t) const; //根据当前状态返回距离“下一事件”的剩余时间（秒）

private:

    //分布
    const SimProcess* m_coldStart;
    const SimProcess* m_coldService;
    const SimProcess* m_warmService;
    const SimProcess* m_expiration;

    //状态与标志
    FunctionState m_state{FunctionState::COLD};
    bool m_reserved{false};
    double m_creation_time{0.0};

    //三个关键 下一事件时间
    double m_next_init_departure{std::numeric_limits<double>::infinity()};
    double m_next_departure{std::numeric_limits<double>::infinity()};
    double m_next_termination{std::numeric_limits<double>::infinity()};

};

#endif //UNTITLED2_FUNCTIONINSTANCE_H