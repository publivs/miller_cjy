#include<stdio.h>
#include<omp.h>

static long counts = 27;
double step ;

int main(int argc,const char* argv[])
{

    long long int i;
    long long int sum = 0;
    long thrdSum[4];

    omp_set_num_threads(4);
    #pragma omp parallel
    {
    int id = omp_get_thread_num();
    thrdSum[id] = 0;
    #pragma omp for
    for(i=1;i<100;i++){
        thrdSum[id] += i;
        }

    for(i=0;i<4;i++){
        sum+=thrdSum[i];
        // printf("value is %d\n",thrdSum[i]);
        }

    }

    printf("result:%d\n",sum);
}