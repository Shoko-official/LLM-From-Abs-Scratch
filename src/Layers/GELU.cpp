#define _USE_MATH_DEFINES
#include "Layers/GELU.hpp"

namespace llm {

    // GELU approximation: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
    Matrix GELU::forward(const Matrix& input) {
        int m = input.m;
        int n = input.n;
        
        last_input = input;
        Matrix output(m, n);

        const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double x = input(i, j);
                double inner = sqrt_2_over_pi * (x + 0.044715 * std::pow(x, 3));
                output(i, j) = 0.5 * x * (1.0 + std::tanh(inner));
            }
        }

        return output;
    }

    Matrix GELU::backward(const Matrix& grad_output) {
        int m = grad_output.m;
        int n = grad_output.n;
        
        Matrix grad_input(m, n);

        const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double x = last_input(i, j);
                double x3 = std::pow(x, 3);
                double inner = sqrt_2_over_pi * (x + 0.044715 * x3);
                double t = std::tanh(inner);
                
                // Derivative approximation
                double sech2 = 1.0 - t * t;
                double deriv = 0.5 * (1.0 + t) + 0.5 * x * sech2 * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
                
                grad_input(i, j) = grad_output(i, j) * deriv;
            }
        }

        return grad_input;
    }

} // namespace llm
