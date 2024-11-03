'''
2NSDU20521004276
'''
import os
import time
import uiautomator2 as u2
# 连接手机#
def connect_phone(device_name):
    d = u2.connect(device_name)
    if not d.service("uiautomator").running():
    # 启动uiautomator服务
        print("start uiautomator")
        d.service("uiautomator").start()
        time.sleep(2)
    if not d.agent_alive:
        print("agent_alive is false")
        u2.connect()
        d = u2.connect(device_name)
    return d
def run(device_name):
    d = connect_phone(device_name)
    d.app_start("com.ccb.longjiLife")

    time_start = time.time()

    while True:
        start = time.time()
        if d(textContains="网点").exists:
            print("点击结算")
            d(textContains="网点").click()
        if d(text="在线取号").exists:
            print("在线取号")
            d(text="在线取号").click()

        if d(text="网点名称").exists:
            print("网点名称")
            d(text="网点名称").click()

        if d(textContains="输入").exists:
            print("输入")
            d(textContains="输入").click()
            d(textContains="全部").click()
            d(textContains="输入").click()
            d.set_fastinput_ime(True)
            d.send_keys("九江路",True)
            d(textContains="全部").click()

        # if d(text="全部").exists:
        #     print("全部")
        #     d(text="全部").click()
        if d(text="确认").exists:
            print("确认")
            d(text="确认").click()

        print("本次花费时间:", time.time() -start)
        print("总共花费时间:", (time.time() -time_start) / 60)

if __name__ == '__main__':
    # 此处填设备编号：由1024我的小表妹原创
    device_name = "2NSDU20521004276"
    run(device_name)


