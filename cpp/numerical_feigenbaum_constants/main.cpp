// compile with -lgmpxx -lgmp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <gmp.h>
#include <gmpxx.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

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

void plus_n(mpf_class& res, mpf_class lambda, mpf_class x)
{
    res = x + lambda;
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

mpf_class point5(mpf_class)
{
    return 0.5;
}

// The first element in A_i must be A_0: the lambda at which the 1-orbit stable fixed point achieves superstability
void cal_feigenbaum(
    void (*map)(mpf_class&, mpf_class, mpf_class),
    mpf_class (*x_bar)(mpf_class),
    std::vector<mpf_class> A_i)
{
    std::vector<mpf_class> delta_cal;
    for (int i = 0; i < A_i.size() - 2; i++) {
        delta_cal.push_back((A_i[i + 1] - A_i[i]) / (A_i[i + 2] - A_i[i + 1]));
    }

    for (int i = 0; i < delta_cal.size(); i++) {
        std::cout << i + 1 << ": " << delta_cal[i] << std::endl;
    }

    std::vector<mpf_class> d_i;
    for (int i = 1; i < A_i.size(); i++) {
        mpf_class local_max = x_bar(A_i[i]);
        mpf_class tmp;
        iterate(tmp, map, std::pow(2, i - 1), A_i[i], local_max);
        d_i.push_back(abs(local_max - tmp));
    }

    std::cout << "delta" << std::endl;
    for (int i = 0; i < d_i.size() - 1; i++) {
        std::cout << d_i[i] / d_i[i + 1] << std::endl;
    }
}

void cal_logistic()
{
    mpz_class counter = 0;
    const mpf_class end = 1;
    const mpf_class alpha = 2.503;
    const mpf_class delta = 4.669;

    mpf_class lambda_plus = 0.28;
    mpf_class increment = 0.001;
    mpf_class lambda(0.7, 128);
    std::vector<mpf_class> A_i = { 0.5 };

    unsigned int cycle_p = 1;
    mpf_class threshold = 0.005;

    while ((lambda < end) && (A_i.size() < 16)) {
        if (is_super_stable(
                logistic,
                [](mpf_class x) -> mpf_class { return 0.5; },
                lambda,
                cycle_p,
                threshold,
                1)) {

            lambda = poke_lambda(logistic, point5, lambda, cycle_p, increment/200, 100);
            std::cout << "lambda found at: " << lambda << std::endl;
            A_i.push_back(lambda);
            increment /= delta;
            lambda_plus /= delta;
            lambda += lambda_plus;
            threshold /= (alpha * 0.9);
            cycle_p += 1;
        }
        lambda += increment;
        counter += 1;
    }
    std::cout << counter << " times iterated!" << std::endl;

    cal_feigenbaum(logistic, [](mpf_class x) -> mpf_class { return 0.5; }, A_i);
}

int main()
{
    std::cout << std::fixed << std::setprecision(10);
    cal_logistic();

    return 0;
}
