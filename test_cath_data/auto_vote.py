import json
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import bs4
import requests
import pytesseract
from PIL import Image
import cv2 as cv
import numpy as np
import random

# C:\Program Files\Tesseract-OCR


chrome_options = Options()
chrome_options.add_argument('window-size=1920x3000') #指定浏览器分辨率
chrome_options.add_argument('--disable-gpu') #谷歌文档提到需要加上这个属性来规避bug
chrome_options.add_argument('--hide-scrollbars') #隐藏滚动条, 应对一些特殊页面
chrome_options.add_argument('--ignore-certificate-errors') #忽略一些莫名的问题
chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])  # 开启开发者模式
chrome_options.add_argument('--disable-blink-features=AutomationControlled')  # 谷歌88版以上防止被检测
# chrome_options.add_argument('blink-settings=imagesEnabled=false') #不加载图片, 提升速度
chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36')

# chrome_options.add_argument('--headless') # 浏览器不提供可视化页面. linux下如果系统不支持可视化不加这条会启动失败,可视化带ui的正常使用,方便调试

def check_proxy():
    url="https://www.baidu.com/"
    ip="114.96.199.198"
    port = "4325"

    proxy={"http":"http://"+ip+":"+port}
    headers={"User-Agent":"Mozilla/5.0"}
    res=requests.get(url,proxies=proxy,headers=headers)
    print(res.status_code)  # 返回200：表示该ip代理是可用的
    print('------------------')

    chrome_path = '/Users/dannihong/downloads/webdriver_browser/chromedriver86'
    options=webdriver.ChromeOptions()
    options.add_argument('--proxy-server=http://'+ip+"："+port)
    options.add_argument('--proxy-server=http://114.96.199.198：4325')  # 必须是中文冒号
    driver=webdriver.Chrome(executable_path=chrome_path, chrome_options=options)
    driver.get(url)

# # 代理池处理
proxy_arr = pool_list = [
    "111.225.152.36:8089",
    "182.139.111.131:9000",
    "121.13.252.62:41564",
    "218.1.200.97:57114",
    "111.225.153.6:8089",
    "171.92.20.187:9000",
    "183.147.209.205:9000",
    "182.139.110.128:9000",
    "111.225.153.216:8089",
    "111.225.153.138:8089",
    "61.171.107.221:9002",
    "171.92.21.83:9000",
    "171.92.21.39:9000",
    "115.220.6.108:8089",
    "115.211.47.95:9000",
    "183.147.208.78:9000",
    "111.225.152.82:8089",
    "111.225.153.66:8089",
    "182.139.110.74:9000",
    "122.243.14.207:9000",
    "106.227.36.124:9002",
    "111.225.152.105:8089",
    "111.225.152.53:8089",
    "171.92.21.75:9000",
    "113.121.36.199:9999",
    "114.116.2.116:8001",
    "117.93.180.175:9000",
    "115.211.42.15:9000",
    "202.109.157.66:9000",
    "111.225.152.88:8089",
    "182.34.17.104:9999"]
ip_port = random.choice(proxy_arr)  # 随机选择一个代理
chrome_options.add_argument(f'--proxy-server=http://{ip_port}')  # 添加代理


driver = webdriver.Chrome(options=chrome_options)  # 将chromedriver放到Python安装目录Scripts文件夹下

options = webdriver.ChromeOptions()

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

def remove_noise(img,k=4):
  img2 = img.copy()
#   img处理数据，k过滤条件
  w,h = img2.shape
  def get_neighbors(img3,r,c):
    count = 0
    for i in [r-1,r,r+1]:
      for j in [c-1,c,c+1]:
        if img3[i,j] > 10:#纯白色
          count+=1
    return count
#   两层for循环判断所有的点
  for x in range(w):
    for y in range(h):
      if x == 0 or y == 0 or x == w -1 or y == h -1:
        img2[x,y] = 255
      else:
        n = get_neighbors(img2,x,y)#获取邻居数量，纯白色的邻居
        if n > k:
          img2[x,y] = 255
  return img2

# Grayscale image
def recognize_captcha(file_path):
    img = Image.open(file_path).convert('L')
    ret,img = cv.threshold(np.array(img), 125, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    # img = remove_noise(img,k=4)
    img = Image.fromarray(img.astype(np.uint8))
    res = pytesseract.image_to_string(img)
    res = res.replace(' ','').replace('\n','')
    print(res)
    return res

















def check_pic_res():
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'lxml')
    # target_table  = soup.find_all('ant-modal-content')
    target_soup = soup.select('body > div.team > form > div > div > div.form-group > div.col-sm-7.col-xs-7.tradition')
    target_table = target_soup[0]
    target_str = target_table.contents[0]
    import re
    rd_str = re.findall(r"randomStr=(.*?)&",str(target_str))[0]
    timestamp = re.findall(r'''timestamp=(.*?)"''',str(target_str))[0]
    req_headers = {
                    'USER_AGENT':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,applicatio6n/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
                    'Connection':'keep-alive',
                    'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8',
                    }
    target_url = f"https://poll.cnfic.com.cn/financier/captcha/getCaptchaImg?randomStr={rd_str}&timestamp={timestamp}"
    res = connect_url(target_url,req_headers)
    res = res.content
    file_path = r"C:\Users\kaiyu\Desktop\miller\test_cath_data\test_1" + ".png"
    playFile = open(file_path, 'wb')
    playFile.write(res)
    playFile.close()

    pic_res = recognize_captcha(file_path)
    print(pic_res)
    return pic_res

def refresh_pic():
    pp_html = driver.page_source
    pic_path = '''/html/body/div[2]/form/div/div/div[2]/div[2]/img'''
    target_pic_btn = driver.find_element(by=By.XPATH, value=pic_path)
    target_pic_btn.click()
    print('pit tick')


driver.get('https://poll.cnfic.com.cn/vote2022/index.html')  # 此处不要再放登录的网址，可以用未登录的首页



driver.refresh()

pic_res = check_pic_res()
pic_res.replace('~','')
refresh_pic()

if ('=?' in pic_res) or ('=7' in pic_res) :
    do_next = 0
    target_str = pic_res[:-2]
    target_str = target_str.replace('=','')
    ret_res = str(eval(target_str))
    next_pic = 0

elif pic_res.__len__() ==4 :
    ret_res = pic_res
    user_input = driver.find_element(by=By.XPATH, value='/html/body/div[2]/form/div/div/div[2]/div[1]/input')
    user_input.send_keys(ret_res)
    next_pic = 0

elif pic_res.__len__() == 0:
    refresh_pic()
    pass


target_path_16 = '''/html/body/div[2]/form/div/div/div[1]/div[16]/div[2]/div/input'''
target_a_btn = driver.find_element(by=By.XPATH, value=target_path_16)
time.sleep(1)
target_a_btn.click()
print('16 is choosed')

target_path_14 = '''/html/body/div[2]/form/div/div/div[1]/div[14]/div[2]/div/input'''
target_b_btn = driver.find_element(by=By.XPATH, value=target_path_14)
time.sleep(1)
target_b_btn.click()
print('14 is choosed')

entry = '''/html/body/div[2]/form/div/div/div[2]/div[4]/button'''
target_3_btn = driver.find_element(by=By.XPATH, value=entry)
target_3_btn.click()

last = '''/html/body/div[2]/form/div/div/div[3]/div/div/div[3]/a'''
target_3_btn = driver.find_element(by=By.XPATH, value=last)
target_3_btn.click()


# cv.imshow('input image', src)
# recognize_text(src)
# print(src.shape)
# cv.waitKey(0)
# cv.destroyAllWindows()










print(1)
# 内容部分
