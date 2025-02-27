#include <gmpxx.h>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
// Approximation of power series with the maximum degree of n, not inclusive
// power series is just a polynomial like 1 + 2x + 3x^2 + 4x^3 + 5x^4 + ...
// This class allow manipulations of power series, like addition, substraction, multiplication, etc
// all terms with degree greater than n are ignored
class power_series_n {
public:
    std::vector<mpf_class> coef;

    power_series_n() { }

    power_series_n(std::vector<mpf_class> coeff);

    power_series_n(long unsigned int num, std::vector<mpf_class> coeff);

    power_series_n(long unsigned int num);

    unsigned long int get_precision() const;

    vector<mpf_class> get_copy_coef() const;

    // the num_of_terms shall be unchanged
    power_series_n operator+(const power_series_n& rhs) const
    {
        auto terms = this->get_precision();
        std::vector<mpf_class> res = this->get_copy_coef();
        for (long unsigned int i = 0; i < terms && i < rhs.get_precision(); i++) {
            res[i] += rhs.coef[i];
        }
        return power_series_n(res);
    }

    // the num_of_terms shall be unchanged
    power_series_n operator*(const power_series_n& rhs) const
    {
        auto terms = this->get_precision();
        std::vector<mpf_class> res(terms, 0);

        for (unsigned int i = 0; i < coef.size(); i++) {
            for (unsigned int j = 0; j < rhs.coef.size(); j++) {
                if (i + j >= terms) {
                    break;
                }
                res[i + j] += coef[i] * rhs.coef[j];
            }
        }

        return power_series_n(terms, res);
    }

    friend power_series_n operator*(double const&, power_series_n const&);

    friend power_series_n operator*(mpf_class const& lhs, power_series_n const& rhs);

    void set_num_of_terms(unsigned int num);

    power_series_n pow(unsigned int n) const
    {
        if (n == 0) {
            vector<mpf_class> tmp = { 1 };
            return power_series_n(tmp);
        }

        power_series_n res = *this;
        for (unsigned int i = 1; i < n; i++) {
            res = res * *this;
        }
        return res;
    }

    std::string to_string() const;

    void print() const;

    power_series_n sub(power_series_n const& in) const
    {
        power_series_n res = power_series_n(this->get_precision());
        for (unsigned int i = 0; i < coef.size(); i++) {
            res = res + coef[i] * in.pow(i);
        }
        return res;
    }
};

std::string power_series_n::to_string() const
{
    std::string res = "";
    res += "(";
    res += std::to_string(this->get_precision());
    res += ") ";

    if (coef.size() == 0) {
        return res;
    }

    bool first_term = true;
    for (unsigned int i = 0; i < coef.size(); i++) {
        if (coef[i] == 0) {
            continue;
        }

        if (coef[i].get_d() < 0) {
            res += " - ";
        } else if (!first_term) {
            res += " + ";
        }
        first_term = false;
        res += std::to_string(abs(coef[i].get_d()));

        if (i == 0) {
        } else if (i == 1) {
            res += "x";
        } else {
            res += "x^";
            res += std::to_string(i);
        }
    }
    return res;
}

power_series_n::power_series_n(std::vector<mpf_class> coeff)
{
    coef = coeff;
}

power_series_n::power_series_n(long unsigned int num, std::vector<mpf_class> coeff)
{
    coef = coeff;
    for (long unsigned int i = num; i < coeff.size(); i++) {
        coef.pop_back();
    }
    for (long unsigned int i = coeff.size(); i < num; i++) {
        coef.push_back(0);
    }
}

power_series_n::power_series_n(long unsigned int num)
{
    coef = vector<mpf_class>(num, 0);
}

unsigned long int power_series_n::get_precision() const
{
    return coef.size();
}

vector<mpf_class> power_series_n::get_copy_coef() const
{
    vector<mpf_class> res = coef;
    return res;
}


void power_series_n::print() const
{
    std::cout << this->to_string() << std::endl;
}

power_series_n operator*(double const& lhs, power_series_n const& rhs)
{
    std::vector<mpf_class> res;
    for (unsigned int i = 0; i < rhs.get_precision(); i++) {
        res.push_back(lhs * rhs.coef[i]);
    }
    return power_series_n(res);
}
power_series_n operator*(mpf_class const& lhs, power_series_n const& rhs)
{
    std::vector<mpf_class> res;
    for (unsigned int i = 0; i < rhs.get_precision(); i++) {
        res.push_back(lhs * rhs.coef[i]);
    }
    return power_series_n(res);
}

void power_series_n::set_num_of_terms(unsigned int num)
{
    if (num < coef.size()) {
        for (unsigned int i = num; i < coef.size(); i++) {
            coef.pop_back();
        }
    } else {
        for (auto i = coef.size(); i < num; i++) {
            coef.push_back(0);
        }
    }
}

int main()
{
    std::vector<mpf_class> v1 = { 1, 2, 3 };
    std::vector<mpf_class> v2 = { 1, 2 };
    power_series_n p1(v1);
    p1.print();
    power_series_n p2(4, v2);
    p2.print();
    // (p1 + p2).print();
    auto p3 = p1.sub(p2);
    p3.print();

    // power_series_n p2 = 2.1 * p1;
    // p2.print();
    // power_series_n p3 = p1 + p2;
    // p3.print();

    return 0;
}
