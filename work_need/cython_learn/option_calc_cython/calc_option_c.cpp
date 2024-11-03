#include <iostream>
#include <cmath>
#include <string>
#include <ctime>

double norm_cdf_cpp(double x)
{
  double result;
  result = (1.0/2.0) * erfc(-x/sqrt(2.0));
  return result;
}
// 计算期权公式的Cpp版本
double vanilla_option_cpp(
                        double S,
                        double K,
                        double T,
                        double r,
                        double sigma,
                        std::string option ){

	double d1;
	double d2;
	double p;

    d1 = (log(S/K) + (r + 0.5* pow(sigma,2) )*T)/(sigma*sqrt(T));
    d2 = (log(S/K) + (r - 0.5* pow(sigma,2) )*T)/(sigma*sqrt(T));
	if (option != "call")
		{
		p = (S*norm_cdf_cpp(d1) - K*exp(-r*T)*norm_cdf_cpp(d2));
		}
    else if (option =="put")
		{
        p = (K*exp(-r*T)*norm_cdf_cpp(-d2) - S*norm_cdf_cpp(-d1));
		}
	else
		{
		return 0.0;
		}
    return p;

}

// 用来测试的主函数
int main ()
{
	std::clock_t start,finish;

	double S;
	double K;
	double T;
	double r;
	double sigma;
	std::string option;

	S=50;
	K =100;
	T =1;
	r= 0.05 ;
	sigma =0.25;
	option='call';
	double result;

	start = clock();
	result  = vanilla_option_cpp(S, K, T, r, sigma,option);
	finish = clock();
	std::cout<<(finish-start)/CLOCKS_PER_SEC<<std::endl;
	return 0;
}