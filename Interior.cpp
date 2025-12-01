#include "Interior.hpp"

std::pair<std::vector<bool>, std::vector<double>> find_interior(const Matrix& A, const std::vector<double>& b) {
    size_t m = A.get_rows();
    size_t n = A.get_cols();
    size_t T_rows = m + 1;
    size_t T_cols = 1 + 2 * n + 2 * m + 1 + 1;

    // Расширенная матрица
    Matrix T(T_rows, T_cols);

    // Заполняем внутренности
    for (size_t i = 1; i < m + 1; i++) {
        for (size_t j = 1; j < n + 1; j++) {
            T.at(i, j) = A(i - 1, j - 1); // Матрица A
            T.at(i, n + j) = -A(i - 1, j - 1); // Матрица -A
        }

        T.at(i, 2 * n + i) = 1; // Единичная матрица slack переменных
        T.at(i, T_cols - 2) = 1; // Стоблец t
    }

    // Заполняем столбец свободных коэффициентов
    // Если он отрицательный, то меняем знак у всей строки
    for (size_t i = 1; i < m + 1; i++) {
        T.at(i, T_cols - 1) = b[i - 1];
        if (b[i - 1] < 0) {
            for (size_t j = 0; j < T_cols; j++) {
                T.at(i, j) *= -1;
            }
        }
    }

    // Заполняем единичную матрицу вспомогательных переменных
    for (size_t i = 1; i < m + 1; i++) {
        T.at(i, 2 * n + m + i) = 1;
    }

    // Заполняем первую строку
    T.at(0, 0) = 1;
    for (size_t i = 1; i < m + 1; i++) {
        T.at(0, 2 * n + m + i) = -1;
        for (size_t j = 0; j < T_cols; j++) {
            T.at(0, j) += T(i, j);
        }
    }

    // Первая часть - ищем решение на краю многогранника
    simplex_routine(T);

    // Если все плохо
    if (std::abs(T(0, T_cols - 1)) > local_eps) return {{false}, {}};

    // Обнуляем искусственные переменные и больше их не трогаем
    for (size_t i = 0; i < T_rows; i++) {
        for (size_t j = 1 + 2 * n + m; j < T_cols - 2; j++) {
            T.at(i, j) = 0;
        }
    }

    // Обновляем ряд целевой функции
    for (size_t j = 1; j < T_cols - 2; j++) {
        T.at(0, j) = 0;
    }
    T.at(0, T_cols - 2) = 1;

    // Если стоблец t - базисный, то приводим матрицу к каноническому виду
    std::vector<size_t> basics = find_basics(T);
    for (size_t i = 0; i < basics.size(); i++) {
        if (basics[i] == T_cols - 2) {
            for (size_t j = 0; j < T_cols; j++) {
                T.at(0, j) -= T(i + 1, j);
            }
        }
    }

    // Вторая часть - ищем решение внутри многогранника
    bool unbounded = simplex_routine(T);
    
    // Ищем базисные столбцы и решение
    std::vector<double> v(2 * n + m + m + 1, 0.0);
    basics = find_basics(T);
    for (size_t i = 0; i < basics.size(); i++) {
        v[basics[i] - 1] = T(i + 1, T_cols - 1);
    }

    // Выражаем искомые переменные: x_i = x_i^+ - x_i^-
    std::vector<double> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = v[i] - v[n + i];
    }

    return {{true, unbounded}, x};
}

bool simplex_routine(Matrix& T) {
    size_t T_rows = T.get_rows();
    size_t T_cols = T.get_cols();

    while (true) {
        // Выбор входящей переменной
        size_t ent = 0;
        double ent_max = 0;
        for (size_t j = 1; j < T_cols - 1; j++) {
            if (T(0, j) <= 0) continue;
            if (T(0, j) > ent_max) {
                ent_max = T(0, j);
                ent = j;
            }
        }

        // Если не нашлось ни одного положительного коэффициента, то выходим
        if (ent == 0) return false;

        // Выбор выходящей переменной
        size_t ext = 0;
        double ext_min = std::numeric_limits<double>::max();
        for (size_t i = 1; i < T_rows; i++) {
            if (T(i, ent) <= 0) continue;
            double t = T(i, T_cols - 1) / T(i, ent);
            if (t < ext_min) {
                ext_min = t;
                ext = i;
            }
        }

        // Вот это я не до конца понял, надо разобраться, но работает
        // Если не смогли найти выходящую переменную, то допустимая область неограничена
        if (ext == 0) {
            // Сжимаем ограничения
            for (size_t i = 1; i < T_rows; i++) {
                T.at(i, T_cols - 1) -= T(i, ent);
            }

            return true;
        }

        // Поворот
        for (size_t i = 0; i < T_rows; i++) {        
            double k = T(i, ent) / T(ext, ent);
            if (i == ext) k -= 1 / T(ext, ent);
            for (size_t j = 0; j < T_cols; j++) {
                T.at(i, j) -= T(ext, j) * k;
            }
        }
    }
}

std::vector<size_t> find_basics(const Matrix& T) {
    size_t T_rows = T.get_rows();
    size_t T_cols = T.get_cols();

    std::vector<size_t> basics(T_rows - 1, 0);
    for (size_t j = 1; j < T_cols - 1; ++j) {

        size_t count_nonzero = 0;
        size_t pivot_row = 0;
        bool unit = true;

        for (size_t i = 1; i < T_rows; ++i) {
            if (std::abs(T(i, j)) > local_eps) {
                count_nonzero++;
                pivot_row = i;
                if (std::abs(T(i, j) - 1.0) > local_eps) unit = false;
            }
        }

        if (count_nonzero == 1 && unit) {
            basics[pivot_row - 1] = j;
        }
    }

    return basics;
}
