C++的OpenMP

基本使用:

    openMP没办法调度Debug

    g++ -fopenmp "输入的文件路径" -o '输出的文件名'.exe

模块基本特性:

1、共享内存，基于线程的并行
OpenMP基于共享内存和多线程，进程由多个线程组成并共享内存。
2、显式并行
OpenMP是一个显式的（不是自动的）编程模型，它为程序员提供了对并行化的完全控制
3、Fork - Join模型

类似于Python的并行化

一个进程的主线程开始，主线程按顺序执行，直到遇到第一个并行区域构造。
b) FORK：主线程创建一组并行线程
c) 程序中由并行区域构造封闭的语句在不同的组线程之间并行执行
d) JOIN：当组线程完成并行区域构造中的语句时，它们将同步并终止，只留下主线程

并行区域(Parallel Region)概念:

1. 并行区域是由多个线程执行的代码块，是基本的OpenMP并行结构。
2. 当一个线程到达并行指令时，它将创建一组线程并成为主（master）线程，其线程号为0。
3. 从这个并行区域开始，代码被复制，所有线程都将执行该代码。
4. 在并行区域的结尾有一个隐含的屏障（barrier），超过此点后只有主线程继续执行。
5. 如果任何线程在一个并行区域内终止，那么该组的所有线程都将终止。

并行域的设置:

线程数量控制:

**
    omp_set_dynamic**(**1**)**;**//启用动态分配线程数
**
    omp_set_num_threads**(**6**)**;**//设置线程数为6个

线程设置是否允许嵌套:

    omp_set_nested(1/0) //设置嵌套或者子线程

IF子句:
    OpenMP中只有 if `<expression> == True会触发并创建子线程`

FOR语法[循环结构的构造\]

1、schedule static子句

pragma omp for schedule(staic,[,chunk]) :将chunk的迭代块静态分配给每个线程

2、schedfule dynamic 子句

#pragma omp parallel for schedule(dynamic [,chunk]):为多个并行的线程动态分配迭代量，这个爹带两保证低于Chunk指定的尺寸

3、schedfule guided子句

**#pragma omp parallel for schedule(guided [,chunk])**

特性:

* chunk的尺寸随着每个分派的迭代量以(接近)指数的方式递减。 除了最后一块除外,各个并行线程依次获得每个连续chunk的块尺寸逐渐减少。
* 分配的迭代量不低于chunk指定的尺寸。
* 开销低于dynamic分配方式、具有更良好的负载均衡性。

4、后缀 nowwait子句

    举例:

    #**pragma** omp for schedule(guided, chunk) nowait

特性:

    如果设置了nowait子句，**则在并行循环的结束处不会进行线程同步，先执行完循环的线程会继续执行后续代码**。

Section和Single

Section

举个例子

```
指定块进行操作1、Section指令 运行进行分块	int nLoopNum = 10;
	int i;#pragma omp parallel shared(nLoopNum) private(i)
	{
		printf("thread %d start\n", omp_get_thread_num());
	#pragma omp sections nowait
		{
			#pragma omp section//看这里
			for (i=0; i < nLoopNum/2; i++)
			{
			printf("section1 thread %d excute i = %d\n", omp_get_thread_num(), i);
			}
			#pragma omp section //看这里
			for (i=nLoopNum/2; i < nLoopNum; i++)
			{
			printf("section2 thread %d excute i = %d\n", omp_get_thread_num(), i);
			}
		}    }
```

指定一个Sections块，里面指定不同的section来进行操作，每个独立的Section都指定一个线程进行操作。

(这里说下,就算并行区域有4个线程，但只有两个section，所以只指定两个随机分配的线程运行section代码)

Single:

强行指定为单线程

```
	int i;#pragma omp parallel private(i)
	{
		printf("thread %d start\n", omp_get_thread_num());
		#pragma omp single
		{
			for (i=0; i < 5; i++)
			{
				printf("section1 thread %d excute i = %d\n", omp_get_thread_num(), i);
			}
		}    }
```

single块里面永远只运行一个线程，强I/O操作的时候可能会用到。

同步结构:

    1、Master(主线程指令)

    Master指令指定的区域只由主线程执行，团队中其他线程都跳过该区域的代码

    2、Critical(指定代码区域)

    Critical线程指令指定的代码区域，一次只能有一个线程执行，当另一个线程到达该区域时，他将阻塞，直到该线程执行完毕退出该区域后执行

    举例

```
`#pragma omp parallel sections
	{
	#pragma omp section
		{
			#pragma omp critical (critical1)
			{
				for (int i=0; i < 5; i++)
				{
					printf("section1 thread %d excute i = %d\n", omp_get_thread_num(), i);
					Sleep(200);
				}
			}		}
	#pragma omp section
		{
			#pragma omp critical (critical2) //(critical2)则是指定不同的Critical,如果是#pragma omp critical (critical1)
			{
				for (int j=0; j < 5; j++)
				{
					printf("section2 thread %d excute j = %d\n", omp_get_thread_num(), j);
					Sleep(200);
				}
			}
		}    }
```

    如上，不同的Section应该分配不同的线程，但是如果不同的Section里面Critical 域的名字一样，那整个Parallel域还是在单线程运行

3、barrier

进程内设置强阻塞

```
 #pragma omp parallel
	{
		printf("thread %d excute first print\n", omp_get_thread_num());
	#pragma omp barrier //看这里
		printf("thread %d excute second print\n", omp_get_thread_num());
	}
```

这是个针对整个并行域的声明,任何并行的线程都必须执行到barrier爲止，线程才会执行后续代码

4、Atomic

- 操作类似于Critical,但是该指令提供了一个最小的关键区，其效率比关键区Critical高。
- 该指令仅适用于紧接其后的单个语句。

```
	int x=0;
#pragma omp parallel num_threads(6)
	{
		for(int i=0; i<100000;++i)
	#pragma omp atomic //注意看这里，只用指定一行就行，效率很快
			x++;
	}  
	printf("%d", x);
```

5、ordered指令

ordered指令指定区域的循环迭代将按串行顺序执行，与单个处理器处理结果顺序一致
ordered指令只能用在for或parallel for中

```
代码示例如下
#pragma omp parallel
	{
	#pragma omp for ordered
		for (int i = 0; i < 10; ++i)
		{
		#pragma omp ordered
			{
				printf("thread %d excute i = %d\n", omp_get_thread_num(), i);
			}
```

多个线程变成了一种奇怪的串行

数据作用域

说明：

- 数据作用域定义了程序串行部分中的数据变量中的哪些以及如何传输到程序的并行部分，定义了哪些变量对并行部分中的所有线程可见，以及哪些变量将被私有地分配给所有线程。
- 数据作用域子句可与多种指令（如parallelL、for、sections等）一起使用，以控制封闭区域变量的作用域。
- OpenMP基于共享内存编程模型，所以大多数变量在默认情况下是共享的。

1、Private

语法示例:

```
	int i = 0;
	float a = 1000.0;
	#pragma omp parallel private(i,a)

	{	code.....}
```

    每个变量i 和a都是独立的，互相不影响，并且，不影响全局的i和a,类似于python中函数的局部变量,每个线程的声明的对应变量都是对新对象的引用;

2、firtprivate&lastprivate

    firstprivate子句指定的变量不仅是private作用范围，同时在进入**并行区域构造前根据其原始对象的值初始化**

    lastprivate子句指定的变量不仅是private作用范围，同时会将**最后一次迭代或最后一个section执行的值**复制回原来的变量

3、shared

- shared子句声明的变量将在所有线程之间共享，所有线程都可以对同一个内存位置进行读写操作
- 程序员需确保多个线程正确地访问共享变量（通过critical atomic等指令）
  示例代码如下

  ```
  int i = 0;
  float a = 512.3;

  #pragma omp parallel shared(i,a)
  	{
  		i = i + 1;
  		printf("thread %d i = %d a= %f\n", omp_get_thread_num(), i, a);
  	}
  	printf("out of parallel i = %d", i);
  	}

  ```

  存在对同一个变量进行高速反复操作的时候,需要注意注意线程的执行情况。

4、default

- default子句允许用户为任何并行区域范围内的**所有变量指定默认的private、shared或none作用范围。**
- 同时可使用private、shared、firstprivate、lastprivate和reduction子句免除特定变量的缺省值

5、规约子句:reduction

- reduction子句对其列表中出现的变量进行规约。
- 首先在并行区域为每个线程创建该变量的私有副本。在并行区域结束时，根据指定的运算将所以线程的私有副本执行规约操作后，赋值给原来的变量中。
- reduction列表中的变量必须命名为**标量变量**，不能是数组或结构类型变量，还必须在并行区域中声明为共享

  **注意，一定得是标量，不能是数组(这点非常非常坑)**

数据类型规定的太死，不是特别好用，批量计算都是数组进数组出

6、ThreadPrivate

这是一个高级的用法。

- threadprivate指令用于将全局变量的副本与线程绑定，即使跨越多个并行区域这种关系也不会改变。
- 该指令必须出现在所列变量声明之后。然后每个线程都会获得自己的变量副本，因此一个线程写入的数据对其他线程不可见。
- 因为是确认了线程中的变量、和全局变量的，所以必须关掉动态线程机制。

```
int  a, b, i, tid;
float x;
#pragma omp threadprivate(a, x)
void TestThreadPrivate()
{
	//关闭动态线程
	omp_set_dynamic(0);
	printf("1st Parallel Region:\n");
#pragma omp parallel private(b,tid)
	{
		tid = omp_get_thread_num();
		a = tid;
		b = tid;
		x = 1.1 * tid +1.0;
		printf("Thread %d:a=%d, b=%d, x= %f\n",tid,a,b,x);
	}  
	printf("2nd Parallel Region:\n");
#pragma omp parallel private(tid)
	{
	  tid = omp_get_thread_num();
	  printf("Thread %d:a=%d, b=%d, x= %f\n",tid,a,b,x);
	}  
}

```
