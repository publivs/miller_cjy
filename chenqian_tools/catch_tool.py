
import time
import json
import requests

def connect_url(target_url,req_headers):
    con_continnue = True
    while con_continnue:
        try:
            res_ = requests.get(target_url,headers=req_headers)
            if res_ is not None:
                con_continnue = False
            else:
                time.sleep(5)
                res_ = requests.get(target_url,headers=req_headers)
        except Exception as e:
            print("链接,出异常了！")
    return res_