#include <cmath>
#include <iostream>

#include "Layers/GELU.hpp"
#include "Layers/ReLU.hpp"
#include "Layers/Sigmoid.hpp"
#include "Layers/Tanh.hpp"

using llm::GELU;
using llm::ReLU;
using llm::Sigmoid;
using llm::Tanh;

/*
 * Activation gradient checks
 * Finite-difference vs backward pass
 */

static double dot_loss(const Matrix& y, const Matrix& dy) {
    double out = 0.0;
    for (int i = 0; i < y.m; i++) {
        for (int j = 0; j < y.n; j++) {
            out += y(i, j) * dy(i, j);
        }
    }
    return out;
}

static Matrix make_input_no_kink() {
    Matrix x(2, 3);
    x(0, 0) = -1.5;
    x(0, 1) = -0.8;
    x(0, 2) = 0.5;
    x(1, 0) = 1.2;
    x(1, 1) = 2.0;
    x(1, 2) = -2.3;
    return x;
}

static Matrix make_grad_output() {
    Matrix dy(2, 3);
    dy(0, 0) = 0.3;
    dy(0, 1) = -1.1;
    dy(0, 2) = 0.7;
    dy(1, 0) = -0.2;
    dy(1, 1) = 1.4;
    dy(1, 2) = -0.9;
    return dy;
}

static void print_grad_res(const char* label, bool ok, double max_diff, double tol) {
    if (ok) {
        std::cout << "[PASS] " << label << " max |numeric - analytic| = " << max_diff << "\n";
    } else {
        std::cout << "[FAIL] " << label << " max |numeric - analytic| = " << max_diff
                  << " (tol=" << tol << ")\n";
    }
}

static bool check_sigmoid() {
    Sigmoid layer;
    Matrix x = make_input_no_kink();
    Matrix dy = make_grad_output();

    layer.forward(x);
    Matrix dx = layer.backward(dy);

    double eps = 1e-6;
    double tol = 1e-5;
    double max_diff = 0.0;

    for (int i = 0; i < x.m; i++) {
        for (int j = 0; j < x.n; j++) {
            Matrix xp = x;
            Matrix xm = x;
            xp(i, j) += eps;
            xm(i, j) -= eps;

            double lp = dot_loss(layer.forward(xp), dy);
            double lm = dot_loss(layer.forward(xm), dy);
            double num = (lp - lm) / (2.0 * eps);

            double diff = std::abs(num - dx(i, j));
            if (diff > max_diff) max_diff = diff;
        }
    }

    bool ok = max_diff <= tol;
    print_grad_res("Sigmoid", ok, max_diff, tol);
    return ok;
}

static bool check_tanh() {
    Tanh layer;
    Matrix x = make_input_no_kink();
    Matrix dy = make_grad_output();

    layer.forward(x);
    Matrix dx = layer.backward(dy);

    double eps = 1e-6;
    double tol = 1e-5;
    double max_diff = 0.0;

    for (int i = 0; i < x.m; i++) {
        for (int j = 0; j < x.n; j++) {
            Matrix xp = x;
            Matrix xm = x;
            xp(i, j) += eps;
            xm(i, j) -= eps;

            double lp = dot_loss(layer.forward(xp), dy);
            double lm = dot_loss(layer.forward(xm), dy);
            double num = (lp - lm) / (2.0 * eps);

            double diff = std::abs(num - dx(i, j));
            if (diff > max_diff) max_diff = diff;
        }
    }

    bool ok = max_diff <= tol;
    print_grad_res("Tanh", ok, max_diff, tol);
    return ok;
}

static bool check_gelu() {
    GELU layer;
    Matrix x = make_input_no_kink();
    Matrix dy = make_grad_output();

    layer.forward(x);
    Matrix dx = layer.backward(dy);

    double eps = 1e-6;
    double tol = 2e-5;
    double max_diff = 0.0;

    for (int i = 0; i < x.m; i++) {
        for (int j = 0; j < x.n; j++) {
            Matrix xp = x;
            Matrix xm = x;
            xp(i, j) += eps;
            xm(i, j) -= eps;

            double lp = dot_loss(layer.forward(xp), dy);
            double lm = dot_loss(layer.forward(xm), dy);
            double num = (lp - lm) / (2.0 * eps);

            double diff = std::abs(num - dx(i, j));
            if (diff > max_diff) max_diff = diff;
        }
    }

    bool ok = max_diff <= tol;
    print_grad_res("GELU", ok, max_diff, tol);
    return ok;
}

static bool check_relu_smooth_points() {
    ReLU layer;
    Matrix x = make_input_no_kink();
    Matrix dy = make_grad_output();

    layer.forward(x);
    Matrix dx = layer.backward(dy);

    double eps = 1e-6;
    double tol = 1e-5;
    double max_diff = 0.0;

    for (int i = 0; i < x.m; i++) {
        for (int j = 0; j < x.n; j++) {
            Matrix xp = x;
            Matrix xm = x;
            xp(i, j) += eps;
            xm(i, j) -= eps;

            double lp = dot_loss(layer.forward(xp), dy);
            double lm = dot_loss(layer.forward(xm), dy);
            double num = (lp - lm) / (2.0 * eps);

            double diff = std::abs(num - dx(i, j));
            if (diff > max_diff) max_diff = diff;
        }
    }

    bool ok = max_diff <= tol;
    print_grad_res("ReLU (smooth points)", ok, max_diff, tol);
    return ok;
}

static bool check_relu_kink_zero() {
    ReLU layer;

    Matrix x(1, 5);
    x(0, 0) = -2.0;
    x(0, 1) = -1e-12;
    x(0, 2) = 0.0;
    x(0, 3) = 1e-12;
    x(0, 4) = 2.0;

    Matrix y = layer.forward(x);

    Matrix dy(1, 5);
    dy(0, 0) = 1.0;
    dy(0, 1) = 2.0;
    dy(0, 2) = 3.0;
    dy(0, 3) = 4.0;
    dy(0, 4) = 5.0;

    Matrix dx = layer.backward(dy);

    bool ok_forward = true;
    ok_forward = ok_forward && (y(0, 0) == 0.0);
    ok_forward = ok_forward && (y(0, 1) == 0.0);
    ok_forward = ok_forward && (y(0, 2) == 0.0);
    ok_forward = ok_forward && (y(0, 3) == x(0, 3));
    ok_forward = ok_forward && (y(0, 4) == x(0, 4));

    bool ok_backward = true;
    ok_backward = ok_backward && (dx(0, 0) == 0.0);
    ok_backward = ok_backward && (dx(0, 1) == 0.0);
    ok_backward = ok_backward && (dx(0, 2) == 0.0);
    ok_backward = ok_backward && (dx(0, 3) == dy(0, 3));
    ok_backward = ok_backward && (dx(0, 4) == dy(0, 4));

    Matrix x0(1, 1);
    x0(0, 0) = 0.0;

    Matrix g(1, 1);
    g(0, 0) = 7.0;

    double eps = 1e-6;
    Matrix xp = x0;
    Matrix xm = x0;
    xp(0, 0) += eps;
    xm(0, 0) -= eps;

    double f0 = dot_loss(layer.forward(x0), g);
    double fp = dot_loss(layer.forward(xp), g);
    double fm = dot_loss(layer.forward(xm), g);

    double left = (f0 - fm) / eps;
    double right = (fp - f0) / eps;

    bool ok_one_sided = std::abs(left - 0.0) < 1e-8 && std::abs(right - 7.0) < 1e-6;

    bool ok = ok_forward && ok_backward && ok_one_sided;

    if (ok) std::cout << "[PASS] ReLU boundary behavior\n";
    else std::cout << "[FAIL] ReLU boundary behavior\n";

    return ok;
}

int main() {
    bool all_ok = true;

    all_ok = check_relu_kink_zero() && all_ok;
    all_ok = check_sigmoid() && all_ok;
    all_ok = check_tanh() && all_ok;
    all_ok = check_relu_smooth_points() && all_ok;
    all_ok = check_gelu() && all_ok;

    if (!all_ok) {
        std::cout << "Activation gradient checks failed.\n";
        return 1;
    }

    std::cout << "All activation gradient checks passed.\n";
    return 0;
}
