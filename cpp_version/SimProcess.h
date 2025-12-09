//
// Created by Xueyuan Huang on 14/10/25.
//

#ifndef UNTITLED2_SIMPROCESS_H
#define UNTITLED2_SIMPROCESS_H

#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>

// 抽象基类
class SimProcess {
public:
    virtual ~SimProcess() = default;
    virtual double sample() const = 0;
};

// 常数分布模拟器
class ConstSimProcess: public SimProcess {
public:
    explicit ConstSimProcess(double rate): m_rate(rate) {
        if (m_rate <= 0.0)
            throw std::invalid_argument("ConstSimProcess: rate must be > 0");
    }
    double sample() const override {return 1.0 / m_rate;}

private:
    double m_rate;
};

// 指数分布模拟器
class ExpSimProcess : public SimProcess {
public:
    explicit ExpSimProcess(double rate, unsigned seed = std::random_device{}()):
    m_rate(rate), m_gen(seed), m_dist(rate) {
        if (m_rate <= 0.0)
            throw std::invalid_argument("ExpSimProcess: rate must be > 0");
    }

    double sample() const override {
        return m_dist (const_cast<std::mt19937&>(m_gen));
    }
private:
    double m_rate;
    mutable std::mt19937 m_gen;
    mutable std::exponential_distribution<double> m_dist;
};

//pareto 分布
class ParetoSimProcess : public SimProcess {
public:
    explicit ParetoSimProcess(double shape, double scale = 1.0, unsigned seed = std::random_device{}()):
    m_shape(shape), m_scale(scale), m_gen(seed) {
        if (m_shape <= 0.0 || m_scale <= 0.0 ){
            throw std::invalid_argument("ParetoSimProcess: shape and scale must be > 0");
        }
    }

    double sample() const override {
        std::uniform_real_distribution<double> u(0.0,1.0);
        double x = u(const_cast<std::mt19937&>(m_gen));
        return m_scale / std::pow(1.0-x, 1.0/m_shape);
    }
private:
    double m_shape;
    double m_scale;
    mutable std::mt19937  m_gen;
};

class GaussianSimProcess : public SimProcess {
public:
    GaussianSimProcess(double rate, double std, unsigned seed = std::random_device{}()):
    m_rate(rate), m_std(std), m_gen(seed) {
        if (m_rate <= 0.0 || m_std < 0.0){
            throw std::invalid_argument("GaussianSimProcess: invalid rate or stddev");
        }
    }

    double sample() const override {
        std::normal_distribution<double> n(1.0 / m_rate, m_std);
        double x = n(const_cast<std::mt19937&>(m_gen));
        return (x < 0.0) ? 0.0 : x;
    }
private:
    double m_rate;
    double m_std;
    mutable std::mt19937 m_gen;
};

#endif //UNTITLED2_SIMPROCESS_H
