import itchat
import pandas as pd
# 登录
from itchat.content import *


@itchat.msg_register([TEXT,NOTE])
def text_reply(msg):
    print(msg)


@itchat.msg_register([TEXT,NOTE],isGroupChat=True)
def daily_task_msg(msg):
    response = msg
    return '收到'
# 登录
itchat.auto_login(hotReload=True)
# 开始接收微信消息
itchat.run()