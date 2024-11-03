#include <omp.h>
#include <stdio.h>
#include <vector>

static long counts = 20;
double step;



#define THREAD_NUMS 5
int main(int argc, const char* argv[]){
    std::vector<double> a ;
    std::vector<double> b ;
    for (size_t i=0;i<counts;i++)
        {
        a.push_back(i);
        b.push_back(i);
        }

    double start = omp_get_wtime();
    int N = a.size();

    #pragma omp parallel for schedule(static,3) shared(a)
    for(size_t i=0;i<N;i++)
    {
        a[i] = a[i]+b[i];
        printf("线程id:%d,a[%d]+b[%d] = %g\n",omp_get_thread_num(),i,i,a[i]);

    }
    double time = omp_get_wtime() -start;
    printf("计算耗时 %.4gs\n",time);
}
