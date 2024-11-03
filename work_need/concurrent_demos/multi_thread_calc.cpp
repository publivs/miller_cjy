#include <omp.h>
#include <stdio.h>
#include <vector>
#include<windows.h>
static long counts = 48;
double step;

#define THREAD_NUMS 6

double func_1(double x){
        double y;
        int rand_num;
        y = x*x;
        // rand_num = rand()%(3000-2000+1) +2000;
        Sleep(2*1000);
        return y;
}

int main(int argc, const char* argv[]){
    std::vector<double> a ;
    std::vector<double> b ;
    std::vector<double> c ;
    for (size_t i=0;i<counts;i++)
        {
        a.push_back(i);
        b.push_back(i);
        }

    double start = omp_get_wtime();
    int N = a.size();

    {
    #pragma omp parallel for schedule(dynamic,THREAD_NUMS)  // 数组计算必须得用动态化的
    for(size_t i=0;i<N;i++)
    {
    double ret_value;
    ret_value = func_1(a[i]+b[i]); // 计算调度用多线程解锁，
    // printf("算子值,%.4g,%.4g %.4g \n",a[i],b[i],ret_value);

    #pragma omp critical (push_in) // 数据提交用barrier/critical上锁
        {
        c.push_back(ret_value);
        // printf("线程id:%d,a[%d]+b[%d] = %.4g\n",omp_get_thread_num(),i,i,c[i]);
        }
    }
    }

    double time = omp_get_wtime() -start;
    printf("计算耗时 %.4gs\n",time);
    printf("length is %d",c.size());
    for (size_t i=0;i<48;i++)
    {
        printf("%.4g \n",c[i]);
    }

}
