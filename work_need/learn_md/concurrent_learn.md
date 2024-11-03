Asyncio

简介


asyc的协程

**意味着不要将宝贵的CPU时间浪费在一个被I/O等待的任务,事件循环通过不断轮询任务队列，以确保立即调度并运行一个处于非I/O等待的任务** 。


这是例子

```python3
import asyncio
import time
import random

start=time.time()

def take_time():
    return "%1.2f秒" % (time.time()-start)

async def task_A():
    print("运行task_A")
    await asyncio.sleep(random.uniform(1.0,8.0)/10)
    print(f"task_A结束!!耗时{take_time()}")

async def task_B():
    print("运行task_B")
    await asyncio.sleep(random.uniform(1.0,8.0)/10)
    print(f"task_B结束!!耗时{take_time()}")

async def task_C():
    print("运行task_C")
    await asyncio.sleep(random.uniform(1.0,8.0)/10)
    print(f"task_C结束!!耗时{take_time()}")


async def task_exect():
    tasks=[task_A(),task_B(),task_C()]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(task_exect())
```


运行和逻辑顺序

1、async是一个函数装饰器。

2、tasks是一个等待执行的函数对象列表。

3、await为跳出线程。

过程就是在Python执行到任意一**个携程函数内部的await关键字**所在代码语句时,并且处于I/O挂起，携程函数的上下文切换就会发生，[返回到上层]

从I/O等待的携程函数会将控制权归还给事件循环，并由事件循环分配给其他就绪的协程函数，反之则不会发生上下文切换


async/await注意事项
