// compile with -lgmpxx -lgmp
#include <algorithm>
#include <bits/stdc++.h>
#include <chrono>
#include <cmath>
#include <gmp.h>
#include <gmpxx.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#define PRECISION 4096 // in bits

using namespace std;

typedef void (*iter_map)(mpf_class&, mpf_class, mpf_class);
typedef mpf_class (*mpf_func)(mpf_class);

const mpf_class ALPHA = mpf_class(2.503, PRECISION);
const mpf_class DELTA = mpf_class(4.669, PRECISION);

template <typename T>
void display(vector<T>& dis)
{
    for (auto i : dis) {
        cout << i << endl;
    }
    cout << "length: " << dis.size() << endl;
}

mpf_class random_mpz(long low, long high)
{
    if (high <= low) {
        throw std::invalid_argument("high must be greater than low in random_mpz!");
    }
    static gmp_randclass r1(gmp_randinit_default);
    static int init = 0;
    if (!init) {
        r1.seed(std::chrono::system_clock::now().time_since_epoch().count());
        init = 1;
    }

    return low + r1.get_f() * (high - low);
}

// ip stands for in place
void logistic(mpf_class& res, mpf_class lambda, mpf_class x)
{
    res = 4 * lambda * x * (1 - x);
}

void logistic_skewed(mpf_class& res, mpf_class lambda, mpf_class x)
{
    res = lambda * x * (1 - x) * (1 - x);
}

// Iterate the functions n times, with input x and parameter lambda,
// and set the res to the result.
// the input map must have signature like logistic_ip
// that is,
// void map(mpf_class &res, mpf_class lambda, mpf_class x)
void iterate(
    mpf_class& res,
    void (*map)(mpf_class&, mpf_class, mpf_class),
    unsigned int n,
    mpf_class lambda,
    mpf_class x)
{
    res = x;
    for (unsigned int i = 0; i < n; i++) {
        map(res, lambda, res);
    }
}

bool is_super_stable(
    void (*map)(mpf_class&, mpf_class, mpf_class),
    mpf_class (*x_bar)(mpf_class),
    mpf_class lambda,
    unsigned int orbit_2_power, // at lambda, map has a stable 2^orbit_2_power orbit.
                                // This means we iterate the map 2^{orbit_2_power} times
    mpf_class threshold,
    unsigned int extra_iter)
{
    mpf_class x = x_bar(lambda);
    mpf_class expected = x_bar(lambda);

    iterate(x, map, std::pow(2, orbit_2_power + extra_iter), lambda, x);
    if (abs(x - expected) < threshold) {
        return true;
    }
    return false;
}

mpf_class poke_lambda(
    void (*map)(mpf_class&, mpf_class, mpf_class),
    mpf_class (*x_bar)(mpf_class),
    mpf_class lambda,
    unsigned int orbit_2_power, // at lambda, map has a stable 2^orbit_2_power orbit.
                                // This means we iterate the map 2^{orbit_2_power} times
    mpf_class poke_delta,
    unsigned int poke_times)
{
    const mpf_class expected = x_bar(lambda);

    // create a list of finer lambda iterate 2**orbit_2_power times.
    // record their difference in abs in diff vector, and pick the lambda corresponds to the smallest one
    std::vector<mpf_class> lambda_list;
    for (int i = 0; i < poke_times; i++) {
        lambda_list.push_back(lambda - poke_delta * i);
    }
    for (int i = 1; i < poke_times; i++) {
        lambda_list.push_back(lambda + poke_delta * i);
    }

    std::vector<mpf_class> diff;
    for (mpf_class val : lambda_list) {
        mpf_class x = expected;
        iterate(x, map, std::pow(2, orbit_2_power), val, x);
        diff.push_back(abs(x - expected));
    }

    auto it = std::min_element(std::begin(diff), std::end(diff));
    return lambda_list[std::distance(std::begin(diff), it)];
}

mpf_class poke_lambda_iterate(
    void (*map)(mpf_class&, mpf_class, mpf_class),
    mpf_class (*x_bar)(mpf_class),
    mpf_class lambda, // poke the lambda value around this
    unsigned int orbit_2_power, // at lambda, map has a stable 2^orbit_2_power orbit.
                                // This means we iterate the map 2^{orbit_2_power} times
    mpf_class poke_delta,
    unsigned int poke_times,
    unsigned int iter_n)
{
    for (unsigned int i = 0; i < iter_n; i++) {
        lambda = poke_lambda(map, x_bar, lambda, orbit_2_power, poke_delta, poke_times);
        poke_delta /= 2;
    }
    return lambda;
}

mpf_class point5(mpf_class)
{
    return 0.5;
}

// The first element in A_i must be A_0: the lambda at which the 1-orbit stable fixed point achieves superstability
void cal_and_display_fei_constants(
    void (*map)(mpf_class&, mpf_class, mpf_class),
    mpf_class (*x_bar)(mpf_class),
    std::vector<mpf_class> A_i)
{
    // calculating alphas
    std::vector<mpf_class> alpha_cal;
    for (int i = 0; i < A_i.size() - 2; i++) {
        alpha_cal.push_back((A_i[i + 1] - A_i[i]) / (A_i[i + 2] - A_i[i + 1]));
    }

    // deltas
    std::vector<mpf_class> d_i;
    for (int i = 1; i < A_i.size(); i++) {
        mpf_class local_max(x_bar(A_i[i]), PRECISION);
        mpf_class tmp;
        iterate(tmp, map, std::pow(2, i - 1), A_i[i], local_max);
        d_i.push_back(abs(local_max - tmp));
    }

    cout << "di" << endl;
    display(d_i);
    cout << "alpha" << endl;
    display(alpha_cal);

    std::vector<mpf_class> delta_cal;
    for (int i = 0; i < d_i.size() - 1; i++) {
        delta_cal.push_back(d_i[i] / d_i[i + 1]);
    }
    std::cout << "delta" << std::endl;
    display(delta_cal);
}

mpf_class one_third(mpf_class)
{
    return mpq_class(1, 3);
}

void find_feigenbaum_numerically(
    iter_map map,
    mpf_func max_func,
    mpf_class lambda_start,
    mpf_class end,
    mpf_class lambda_jump,
    mpf_class lambda_inc,
    vector<mpf_class> A_i,
    unsigned int start_cycle,
    mpf_class threshold,
    unsigned int stop_at)
{
    while ((lambda_start < end) && (A_i.size() < stop_at)) {
        if (is_super_stable(
                map,
                max_func,
                lambda_start,
                start_cycle,
                threshold,
                1)) {

            lambda_start = poke_lambda_iterate(map, point5, lambda_start, start_cycle, lambda_inc / 200, 100, 2);
            A_i.push_back(lambda_start);
            lambda_inc /= DELTA;
            lambda_jump /= DELTA;
            lambda_start += (lambda_jump * 0.9);
            threshold /= (ALPHA * 0.9);
            start_cycle += 1;
        }
        lambda_start += lambda_inc;
    }

    cout << "lambda" << endl;
    display(A_i);

    cal_and_display_fei_constants(map, max_func, A_i);
}

int main()
{
    std::cout << std::fixed << std::setprecision(64);

    std::vector<mpf_class> A_i = { 0.5 };
    find_feigenbaum_numerically(
        logistic,
        [](mpf_class) -> mpf_class { return 0.5; },
        mpf_class(0.7, PRECISION),
        mpf_class(1, PRECISION),
        mpf_class(0.28, PRECISION),
        mpf_class(0.001, PRECISION),
        A_i,
        1,
        mpf_class(0.005, PRECISION),
        10);

    std::vector<mpf_class> A_i_skewed = { 2.25 };
    find_feigenbaum_numerically(
        logistic_skewed,
        one_third,
        mpf_class(2.8, PRECISION),
        mpf_class(6, PRECISION),
        mpf_class(2.8, PRECISION),
        mpf_class(0.001, PRECISION),
        A_i_skewed,
        1,
        mpf_class(0.001, 128),
        16);

    return 0;
}
