'''
Cython
'''

#cython:language_level=3

cdef class Fruit(object):
    '''Fruit Type'''

    cdef str name
    cdef double qty
    cdef double price

    def __init__(self,nm,qt,pc):
          self.name=nm
          self.qty=qt
          self.price=pc

    def amount(self):
        return  self.qty*self.price



'''
上面的Cython代码
typedef struct{
    char* name;
    double qty;
    double price;

}    Fruit;

double amount(Fruit* self){
      return self->qty*self->price;
}
'''

'''
Cython中针对结构体的属性会进行访问控制
针对字段:
cdef有三种控制方法
private[默认的]:对于Cython编译器来说，任何使用cdef定义的类属性/方法默认是私有的，
        类外部代码无法访问之,并且在声明类成员时,Cython语法层面不提供显式的private关键字声明，
        因为像cdef private double price 等同画蛇舔足。
public[继承c++的特性，python所有的类本质都是Public]: Cython编译器继承了C++这一特性，
                                            例如类内声明cdef public double price,
                                            表示外部代码可以自由访问和修改该属性值。
protected[方法私有化]:要在Cython中实现类似C++类继承的protected访问控制特性，
                    在Cython中不能使用protected，
                    而是使用cppclass关键字,关于此方面内容以后再说。
'''

'''
针对函数
cdef声明的方法是忽略控制字符的
举个例子:
    cdef public double amount(self)
    想完成pd.add(1,2)，是会报错的
    会告诉你add无法被调用,
    Python外部代码公开C级别的方法是这样
    cpdef double amount (self)
    就可以,编译器是这个操作。
    因为C的类方法是绑定指针的私有方法,外面包一层Python方法即可

    cdef double _amount(self):
         return self.qty * self.price

    def amount(self):
        return return self._amount()
'''