import itchat
from itchat.content import *
import os
import time
from playsound import playsound
import xml.dom.minidom
temp = 'E:/python/微信脚本/消息' + '/' + '撤回的消息'
new_temp = 'E:/python/微信脚本/消息' + '/' + '聊天记录'
tp = r'E:/python/微信脚本/消息/撤回的消息'
ntp = r'E:/python/微信脚本/消息/聊天记录'
dict = {}
 
 
@itchat.msg_register([TEXT, PICTURE, FRIENDS, CARD, MAP, SHARING, RECORDING, ATTACHMENT, VIDEO])
def resever_info(msg):
    global dict
    info = msg['Text']
    msgId = msg['MsgId']
    info_type = msg['Type']
    name = msg['FileName']
    fromUser = itchat.search_friends(userName=msg['FromUserName'])['RemarkName']
    ticks = msg['CreateTime']
    time_local = time.localtime(ticks)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    dict[msgId] = {"info": info, "info_type": info_type, "name": name, "fromUser": fromUser, "dt": dt}
    msg_type = dict[msgId]['info_type']
    if msg_type == 'Text':
        text_info = dict[msgId]['info']
        fromUser = dict[msgId]['fromUser']
        dt = dict[msgId]['dt']
        print('时间:' + dt + '\n' + fromUser + ':' + text_info)
    elif msg_type == 'Picture':
        picture_info = dict[msgId]['info']
        fromUser = dict[msgId]['fromUser']
        dt = dict[msgId]['dt']
        info_name = dict[msgId]['name']
        picture_info(new_temp + '/' + info_name)
        print('时间:' + dt + '\n' + fromUser + ':' + '<图片>')
    elif msg_type == 'Recording':
        recording_info = dict[msgId]['info']
        info_name = dict[msgId]['name']
        fromUser = dict[msgId]['fromUser']
        dt = dict[msgId]['dt']
        recording_info(new_temp + '/' + info_name)
        playsound(new_temp + '/' + info_name)
        print('时间:' + dt + '\n' + fromUser + ':' + '<语音>')
 
 
@itchat.msg_register(NOTE)
def note_info(msg):
    content = msg['Content']
    doc = xml.dom.minidom.parseString(content)
    result = doc.getElementsByTagName("msgid")
    msgId = result[0].childNodes[0].nodeValue
    msg_type = dict[msgId]['info_type']
    if '撤回了一条消息' in msg['Text']:
        if msg_type == 'Recording':
            recording_info = dict[msgId]['info']
            info_name = dict[msgId]['name']
            fromUser = dict[msgId]['fromUser']
            dt = dict[msgId]['dt']
            recording_info(temp + '/' + info_name)
            send_msg = '发送人:' + fromUser + '\n' + '发送时间:' + dt + '\n' + '撤回了一条语音'
            itchat.send(send_msg, 'filehelper')
            itchat.send_file(temp + '/' + info_name, 'filehelper')
            del dict[msgId]
            print(fromUser + " 撤回了一条消息")
        elif msg_type == 'Text':
            text_info = dict[msgId]['info']
            fromUser = dict[msgId]['fromUser']
            dt = dict[msgId]['dt']
            send_msg = '发送人:' + fromUser + '\n' + '发送时间:' + dt + '\n' + '撤回内容:' + text_info
            itchat.send(send_msg, 'filehelper')
            del dict[msgId]
            print(fromUser + " 撤回了一条消息")
        elif msg_type == 'Picture':
            picture_info = dict[msgId]['info']
            fromUser = dict[msgId]['fromUser']
            dt = dict[msgId]['dt']
            info_name = dict[msgId]['name']
            picture_info(temp + '/' + info_name)
            send_msg = '发送人:' + fromUser + '\n' + '发送时间:' + dt + '\n' + '撤回了一张图片'
            itchat.send(send_msg, 'filehelper')
            itchat.send_file(temp + '/' + info_name, 'filehelper')
            del dict[msgId]
            print(fromUser + " 撤回了一条消息")
 
 
def clear_cache(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            clear_cache(c_path)
        else:
            os.remove(c_path)
 
 
if __name__ == '__main__':
    if not os.path.exists(temp):
        os.mkdir(temp)
    if not os.path.exists(new_temp):
        os.mkdir(new_temp)
    clear_cache(tp)
    clear_cache(ntp)
    itchat.auto_login()
    itchat.run()