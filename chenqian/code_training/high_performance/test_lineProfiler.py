import random
import time
from line_profiler import LineProfiler

def function1():
    time.sleep(2)
    for _ in range(100):
        y = random.randint(1,100)

def function2():
    time.sleep(5)
    for _ in range(100):
        y = random.randint(1,100)

def function3():
    time.sleep(7)
    for _ in range(100):
        y = random.randint(1,100)

def main():
    function1()
    function2()
    function3()


if __name__ == "__main__":
    lp = LineProfiler()  # 构造分析对象
    """如果想分析多个函数，可以使用add_function进行添加"""
    lp.add_function(function1)  # 添加第二个待分析函数
    lp.add_function(function2)  # 添加第三个待分析函数
    test_func = lp(main)  # 添加分析主函数，注意这里并不是执行函数，所以传递的是是函数名，没有括号。
    test_func()  # 执行主函数，如果主函数需要参数，则参数在这一步传递，例如test_func(参数1, 参数2...)
    lp.print_stats()  # 打印分析结果
    """
    坑点：
        1：test_func = lp(main)这一步，是实际分析的入口函数（即第一个被调用的函数，但不一定是main函数），所以这里封装的函数必须是测试脚本要执行的第一个函数。
        2：test_func()这一步才是真正执行，如果缺少这一步，代码将不会被执行
        3：lp.print_stats()这一步是打印分析结果，如果缺少这一步，将不会在终端看到分析结果。
        4：分析结果只与是否加入分析队列有关，而不管该函数是否被实际执行，如果该函数加入分析但没有被执行，则Hits、Time等值为空。
    """
