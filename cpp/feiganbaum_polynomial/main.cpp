#include <vector>
#include <iostream>
#include <gmp.h>

// Approximation of power series with the maximum degree of n, not inclusive
// power series is just a polynomial like 1 + 2x + 3x^2 + 4x^3 + 5x^4 + ...
// This class allow manipulations of power series, like addition, substraction, multiplication, etc
// all terms with degree greater than n are ignored
class power_series_n {
    public: 
        unsigned int degree;
	std::vector<mpf_t> coefficients;
};

void foo (mpz_t result, const mpz_t param, unsigned long n) {
    mpz_mul_ui (result, param, n);
    for (unsigned int i = 1; i < n; i++)
        mpz_add_ui (result, result, i*7);
}


int main() {
    mpz_t r, n;
    mpz_init(r);
    mpz_init_set_str(n, "123456129859734821945737651364578126784678123567312628", 0);
    foo(r, n, 20L);
    gmp_printf("Result: %Zd\n", r);
    return 0;
}
