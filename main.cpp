#include <iostream>

#include <cmath>
#include <vector>

#include "LinAlg.hpp"
#include "Interior.hpp"

//===================================================//

// Поиск минимума линейно-квадратичного функционала
// методом внутренней точки

//===================================================//

//===================================================//

// Входные условия

//===================================================//

size_t n = 2;
Matrix Q = Matrix({{0.5, 0}, {0, 0.25}});
std::vector<double> c = {-3, -2};

// Пример: ограниченная прямоугольная область
size_t m = 4;
Matrix A = Matrix({{1, 0}, {0, 1}, {-1, 0}, {0, -1}});
std::vector<double> b = {2, 3, 0, 0};

// Пример: неограниченная область, функционал ограничен снизу
// size_t m = 2;
// Matrix A = Matrix({{-3, -1}, {1, 2}});
// std::vector<double> b = {-5, 0};

// size_t n = 2;
// Matrix Q = Matrix({{-0.5, 0}, {0, -0.25}});
// std::vector<double> c = {2, 2};

// Пример: неограниченная область, функционал не ограничен снизу
// size_t m = 2;
// Matrix A = Matrix({{-3, -1}, {1, 2}});
// std::vector<double> b = {-5, 0};

double mu_0 = 0.1;
double mu_next(double mu_k, unsigned k) {
    // return mu_k / (k + 1);
    return mu_k * 0.999;
}

double eps = 0.0001;

bool is_x0_provided = false;
std::vector<double> x0 = {1, 2};

//===================================================//

// Решение задачи

//===================================================//

double f(const std::vector<double>& x) {
    return x * (Q * x) + c * x;
}

double F(const std::vector<double>& x, double mu) {
    double val_f = f(x);

    double val_barrier = 0;
    std::vector<double> g = A * x - b;
    for (size_t j = 0; j < m; j++) {
        val_barrier -= log(-g[j]);
    }
    val_barrier *= mu;

    return val_f + val_barrier;
}

std::vector<double> gradient(const std::vector<double>& x, double mu) {
    std::vector<double> grad_f = Q * x * 2 + c;

    std::vector<double> grad_barrier(x.size(), 0.0);
    std::vector<double> g = A * x - b;
    for (size_t j = 0; j < m; j++) {
        grad_barrier = grad_barrier - A(j) / g[j];
        std::cout << j << std::endl;
    }
    grad_barrier = grad_barrier * mu;

    return grad_f + grad_barrier;
}

Matrix AtA = A.transpose() * A;
Matrix hessian(const std::vector<double>& x, double mu) {
    Matrix H_f = Q * 2;

    Matrix H_barrier(x.size(), x.size(), 0.0);
    std::vector<double> g = A * x - b;
    for (size_t j = 0; j < m; j++) {
        H_barrier = H_barrier + AtA / (g[j] * g[j]);
    }
    H_barrier = H_barrier * mu;

    return H_f + H_barrier;
}

bool is_in_region(std::vector<double>& x) {
    std::vector<double> g = A * x - b;
    
    for (size_t j = 0; j < m; j++) {
        if (g[j] > 0) return false;
    }

    return true;
}

int main() {

    std::pair<std::vector<bool>, std::vector<double>> point = find_interior(A, b);

    if (is_x0_provided) {
        if (is_in_region(x0)) {
            std::cout << "Предоставлена начальная точка: " << x0 << std::endl;
        } else {
            std::cout << "Предоставлена неверная начальная точка" << std::endl;
            return 0;
        }
    } else {
        if (point.first[0]) {
            x0 = point.second;
            std::cout << "Найдена начальная точка: " << x0 << std::endl;
        } else {
            std::cout << "Нет допустимой области" << std::endl;
            return 0;
        }
    }
    std::cout << std::endl;

    if (point.first[1]) {
        std::cout << "Исследуемая область неограничена" << std::endl;
        std::cout << "Удостоверьтесь, что функция ограничена снизу внутри области" << std::endl;
        std::cout << "Продолжить? (y/n) ";
        char cont;
        std::cin >> cont;
        if (cont == 'n') return 0;
    }

    std::vector<double> x = x0;
    unsigned k = 0;
    double mu = mu_0;
    double F_cur = F(x, mu_0);

    while (true) {
        k++;
        mu = mu_next(mu, k);

        std::vector<double> grad = gradient(x, mu);
        Matrix H = hessian(x, mu);

        double alpha = grad * grad / (grad * (H * grad));

        std::vector<double> x1;
        while (true) {
            x1 = x - grad * alpha;
            if (is_in_region(x1)) break;
            alpha /= 2;

            std::cout << "Шагнули слишком далеко, новый шаг: " << alpha << std::endl;
        }
        x = x1;

        double F_next = F(x, mu);
        double delta = std::abs(F_next - F_cur);

        std::cout << "k = " << k << std::endl;
        std::cout << "x = " << x << std::endl;
        std::cout << "f(x) = " << f(x) << std::endl;
        std::cout << "F(x) = " << F(x, mu) << std::endl;
        std::cout << "delta = " << delta << std::endl;
        std::cout << "mu = " << mu << std::endl;
        std::cout << "grad = " << grad << std::endl;
        std::cout << "alpha = " << alpha << std::endl;
        std::cout << std::endl;

        if (delta < eps) break;
        F_cur = F_next;
    }

    std::cout << "x0 = " << x0 << std::endl;
    std::cout << "k_fin = " << k << std::endl;
    std::cout << "x_opt = " << x << std::endl;
    std::cout << "f(x_opt) = " << f(x) << std::endl;

    return 0;
}