# file: hello.pyx
def say_hello_to(name):
    print("Hello %s!" % name)

cpdef add_func(double a,double b):
    return a+b