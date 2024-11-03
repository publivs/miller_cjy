#include <iostream>

static long long num  = 1000;
double step;

int main(){
    int i = 0;
    double x ,pi,sum = 0;
    step=1.0/(double)num; //这里完成数据转换

    for(i=0;i<num;i++)
        {
        x=(i+0.5)*step;
        sum=sum+4.0/(1.0+x*x);
        }
    pi=step*sum;
    printf("pi=%.16f\n",pi);
}