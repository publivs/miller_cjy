#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <vector>
#include<random>
#include<windows.h>

#define THREAD_NUMS 10
static long counts = 10000;

double func_1(double x){
        double y;
        y = x*x;
        Sleep(1*1000);
        return y;
}

int main()
{
    std::vector<double> input_arr;
    std::vector<double> op_arr;

    for(size_t i = 0;i<counts;i++)
        {
        input_arr.push_back(i);
        }
    printf("vector gene finished...");
    double start =  omp_get_wtime();
    omp_set_num_threads(THREAD_NUMS);
    #pragma parallel for schedule(static,THREAD_NUMS) shared(input_arr) nowait
    {
            std::vector<double> inside_arr;
            double y_i;

            for(size_t i =0;i < input_arr.size();i++)
                {
                int id = omp_get_thread_num();
                y_i = func_1((double)input_arr[i]);
                inside_arr.push_back(y_i);
                printf("value = %f,thread_num = %d \n",y_i,id);
                }
        #pragma omp critical (merge_array)
        {
        op_arr.insert(op_arr.end(),inside_arr.begin(),inside_arr.end());
        }

    }
    int size = op_arr.size();
    printf("size is %d",size);
    double finish = omp_get_wtime() - start;
    printf("计算耗时%.4g\n",time);
    printf("last_one is %d",op_arr[counts-1]);
}

// 为啥我算出来的时单线程,不是多线程并发
