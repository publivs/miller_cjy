
from win32com.client import Dispatch

app =Dispatch('Word.Application')
# 新建word文档
doc = app.Documents.Add()

app.Visible = 1


# https://blog.csdn.net/weixin_43697367/article/details/125202729
