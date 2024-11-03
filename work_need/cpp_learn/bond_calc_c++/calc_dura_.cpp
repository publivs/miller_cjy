#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <limits>
double calculate_ytm(double presentvalue,  
                    const std::vector<double>& cashflow,
                    const std::vector<double>& time_list,
                    double guess = 0.01,
                    double tol = 1e-6,
                    int max_iter = 100)
    {
    std::function<double(double)> ff = [&](double y) {
        double cash_all = 0;
        for (size_t i = 0; i < cashflow.size(); ++i) {
            if (cashflow[i] != cashflow.back()) {
                cash_all += cashflow[i] / std::pow(1 + y, time_list[i]);
            }
        }
        cash_all += cashflow.back() / std::pow(1 + y, time_list.back());
        return cash_all - presentvalue;
    };

    double y = guess;
    double y_old = guess;
    for (int i = 0; i < max_iter; ++i) {
        double f_value = ff(y);
        if (std::abs(f_value) < tol) {
            return y;
        }
        double f_prime = (ff(y) - ff(y_old)) / (y - y_old);
        y_old = y;
        y = y - f_value / f_prime;
    }
    return y;
}

int main() 
{
    double presentvalue = 1000.0;
    std::vector<double> cashflow = {50.0, 50.0, 50.0, 1050.0};
    std::vector<double> time_list = {1.0, 2.0, 3.0, 4.0};
    double y = calculate_ytm(presentvalue, cashflow, time_list);
    if (std::isnan(y) || std::isinf(y)) {
        std::cout << "Yield-to-Maturity is not defined." << std::endl;
    } else {
        std::cout << "Yield-to-Maturity: " << y << std::endl;
    }
    return 0;
}