// main.cpp
// main_serverless.cpp
// Step 5: event-driven timeline to exercise ServerlessSimulator

#include <iostream>
#include <fstream>
#include <chrono>

#include "json.hpp"               // 第三方 json 库（nlohmann）
#include "SimProcess.h"
#include "FunctionInstance.h"
#include "ServerlessSimulator.h"
#include "AutoScalingAlgorithm.h"

int main(int argc, char** argv) {
    using json = nlohmann::json;

    // 1. 选择要读的 JSON 文件：默认用 input_test_tiny.json
     std::string config_file;
    if (argc > 1) {
        // 将来你也可以 ./untitled2 input_example_quick.json 这样切换
        config_file = argv[1];
    }

    std::ifstream in(config_file);
    if (!in) {
        std::cerr << "无法打开配置文件: " << config_file << std::endl;
        return 1;
    }

    json cfg;
    in >> cfg;

    // 2. 从 JSON 中读参数
    // --- 基本速率 ---
    double arrival_rate       = cfg["arrival_rate"];
    double warm_rate          = cfg["warm_service"]["rate"];
    double cold_service_rate  = cfg["cold_service"]["rate"];
    double cold_start_rate    = cfg["cold_start"]["rate"];
    double expiration_rate    = cfg["expiration"]["rate"];

    // --- theta 初始值: [[theta, theta_min, gamma_exp]] ---
    auto theta_json = cfg["theta"][0];
    std::array<double,3> theta_init{
            theta_json[0].get<double>(),
            theta_json[1].get<double>(),
            theta_json[2].get<double>()
    };

    // --- 其它 algorithm 参数 ---
    double tau        = cfg["tau"];
    int    max_conc   = cfg["max_currency"];   // 注意：JSON 里写的是 max_currency
    double T_max      = cfg["max_time"];
    int    K          = cfg["K"];
    double k_delta    = cfg["k_delta"];

    std::array<double,3> k_gamma{
            cfg["k_gamma"][0].get<double>(),
            cfg["k_gamma"][1].get<double>(),
            cfg["k_gamma"][2].get<double>()
    };

    // --- 随机种子：先用第一个 seed ---
    unsigned base_seed = cfg["seeds"][0].get<unsigned>();

    // 跟你以前一样做几组不同的 seed，用来喂给不同的 SimProcess
    unsigned seed_arr  = base_seed ^ 0x9e3779b9u;
    unsigned seed_warm = base_seed ^ 0x85ebca6bu;
    unsigned seed_cold = base_seed ^ 0xc2b2ae35u;
    unsigned seed_cs   = base_seed ^ 0x27d4eb2fu;
    unsigned seed_exp  = base_seed ^ 0x165667b1u;

    // 3. 构造随机过程（这里 tiny 配置里都是 Exponential，我们就用 ExpSimProcess）
    ExpSimProcess arrival(arrival_rate,      seed_arr);
    ExpSimProcess cold_start(cold_start_rate,  seed_cs);
    ExpSimProcess warm_service(warm_rate,      seed_warm);
    ExpSimProcess cold_service(cold_service_rate, seed_cold);
    ExpSimProcess expiration(expiration_rate,    seed_exp);

    // 4. 构造 AutoScalingAlgorithm （用你现在 C++ 里已有的构造函数）
    AutoScalingAlgorithm algo(
            max_conc,         // N
            k_delta,          // k_delta
            k_gamma,          // k_gamma (array<double,3>)
            theta_init,       // theta_init (array<double,3>)
            tau,              // tau
            K,                // K
            T_max             // T_max
    );

    // 5. 构造模拟器
    ServerlessSimulator sim(
            arrival,
            cold_start,
            cold_service,
            warm_service,
            expiration,
            max_conc,
            algo
    );

    // 初始时刻 t0=0，所有池子都为空
    sim.initialize_system(0.0, 0, 0, 0, 0);

    // 6. 运行仿真
    sim.run(false, false);

    // 7. 输出结果
    sim.print_sunmmary();   // 如果你函数名是 print_summary，就改成那个

    return 0;
}
