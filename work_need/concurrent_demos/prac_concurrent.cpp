#include <cstdio>
#include "omp.h"

// int main(int argc, char* argv[])
// {
//     int nthreads, tid;
//     #pragma omp parallel private(nthreads, tid) //{  花括号写在这会报错
//     {
//         tid = omp_get_thread_num();
//         printf("Hello World from OMP thread %d\n", tid);
//         if(tid == 0)
//             {
//             nthreads = omp_get_num_threads();
//             printf("Number of threads %d\n", nthreads);
//             }
//     }
//     return 0;
// }

void foo(int id ,double *a,int size){
        double sum = 0.0;
        for (int i = 0;i<size;i++){
            sum +=a[i];
        }
        printf("线程%d计算的a数组的结果是%f\n",id,sum);
}

int main(){
    double A[1000];
    for(int i =0;i<1000;i++){A[i]=i;}
    //openMP运行的时候函数请求数量明确的线程
    omp_set_num_threads(4);
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        foo(id,&A[0],1000);
    }
}
