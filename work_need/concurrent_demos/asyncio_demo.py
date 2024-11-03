import asyncio
import time
import random

start=time.time()

def take_time():
    return "%1.2f秒" % (time.time()-start)

async def task_A(a):
    print("运行task_A")
    await asyncio.sleep(a)
    print(f"task_A结束!!耗时{take_time()}")

async def task_B(a):
    print("运行task_B")
    await asyncio.sleep(a)
    print(f"task_B结束!!耗时{take_time()}")

async def task_C(a):
    print("运行task_C")
    await asyncio.sleep(a)
    print(f"task_C结束!!耗时{take_time()}")

async def task_exect():
    tasks = [task_A(i) for i in range(3,6)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(task_exect())