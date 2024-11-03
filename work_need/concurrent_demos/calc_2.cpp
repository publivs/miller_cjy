#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <vector>

static long long counts  = 10000000;
double step;
#define THREAD_NUMS 10

int main(int argc,const char* argv[] ){
    double pi = 0.0;
    double start,time,sum;
    start = omp_get_wtime();
    step = 1.0/(double)counts;

    int unit = counts/THREAD_NUMS;
    omp_set_num_threads(THREAD_NUMS);
    #pragma omp parallel
    {
    int id = omp_get_thread_num();
    int left = id*unit+1;
    int right = (id+1)*unit;
    double x,sumPart = 0.0;
    // printf("线程id:%d,数据区间%d~%d\n",id,left,right);

    #pragma omp critical (calc_integration) //critical线程，一次指定一个线程运行此代码
    {
    printf("线程 %d 在运行...\n",id);
    while (left <= right)
        {
        x = (left +0.5)*step;
        {
        sumPart+= 4.0/(1+ x*x);
        }
        left++;
        }
    #pragma omp atomic //对线程里的数据进行原子化操作,涉及到汇总值
        sum += sumPart;
    }
    }
    pi += sum*step;
    time = omp_get_wtime() - start;
    printf("Pi  = %f计算耗时%f线程共%d",pi,time,THREAD_NUMS);
}