#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <type_traits>
double ytm_nolast(double PV, double C, double freq, std::vector<int> d, double n, double M, int TS, std::vector<double> cf = {}) {
    if (cf.empty()) {
        cf.push_back(C / freq);
    }
    
    if (TS == 0) {
        std::cerr << "TS is 0, cannot solve" << std::endl;
        return NAN;
    }
    
    auto f_d = [&](double y) {
        std::vector<double> t(n);
        for (int i = 0; i < n; i++) {
            t[i] = i;
        }
        std::vector<double> coupon(n);
        for (int i = 0; i < n; i++) {
            coupon[i] = cf[i] * pow(1 + y / freq, -((d[i]) / TS + t[i]));
        }
        return std::accumulate(coupon.begin(), coupon.end(), 0.0) + M * pow(1 + y / freq, -((d[n-1]) / TS + n - 1)) - PV;
    };
    
    auto f_d_list = [&](double y) {
        std::vector<double> coupon(n);
        for (int i = 0; i < n; i++) {
            coupon[i] = cf[i] * pow(1 + y, -((d[i]) / TS));
        }
        return std::accumulate(coupon.begin(), coupon.end(), 0.0) - PV;
    };
    
    auto f_diff = [&](double y) {
        double delta_y = 0.000001;
        return (f_d(y + delta_y) - f_d(y - delta_y)) / (2 * delta_y);
    };
    
    auto f = f_d;
    if (std::is_same_v<decltype(d), std::vector<int>>) {
        f = f_d_list;
    }
        
    double y_guess = C / 100.0;
    int maxiter = 50;
    double tol = 1e-8;
    for (int i = 0; i < maxiter; i++) {
        double y_old = y_guess;
        y_guess = y_guess - f(y_guess) / f_diff(y_guess);
        if (std::abs(y_guess - y_old) < tol) {
            return y_guess;
        }
    }
    
    std::cerr << "Failed to solve" << std::endl;
    return NAN;
}
int main() {
    double PV = 90.9743;
    double C = 6.0;
    double freq = 1.0;
    std::vector<int> d = {185, 550};
    double n = 2.0;
    double M = 0.0;
    int TS = 365;
    std::vector<double> cf = {16.0, 95.4};
    double YTM = ytm_nolast(PV, C, freq, d, n, M, TS, cf);
    std::cout << "The yield to maturity of the bond is: " << YTM * 100 << "%" << std::endl;
    return 0;
}