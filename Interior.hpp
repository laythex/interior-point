#pragma once

#include <limits>

#include "LinAlg.hpp"

const double local_eps = 1e-10;

std::pair<std::vector<bool>, std::vector<double>> find_interior(const Matrix& A, const std::vector<double>& b);

bool simplex_routine(Matrix& T);

std::vector<size_t> find_basics(const Matrix& T);