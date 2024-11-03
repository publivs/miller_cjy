
# 纯Python版本
class Fruit(object):
    '''Fruit Type'''
    def __init__(self,nm,qt,pc):
          self.name=nm
          self.qty=qt
          self.price=pc

    def amount(self):
          return  self.qty*self.price




%%cython
class Fruit(object):
    '''Fruit Type'''
    def __init__(self,nm,qt,pc):
          self.name=nm
          self.qty=qt
          self.price=pc

    def amount(self):
          return  self.qty*self.price

%load_ext cython
%%cython
cdef class Fruit(object):
      '''
      这里变量声明的时候一定要
      '''
      cdef str name
      cdef double qty
      cdef double price

      def __init__(self,nm,qt,pc) -> None:
            self.name = nm
            self.qty = qt
            self.price = pc

      def amount(self):
            return self.qty * self.price