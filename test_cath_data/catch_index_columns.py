import json
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

import pandas as pd

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

driver = webdriver.Chrome(options=chrome_options)  # 将chromedriver放到Python安装目录Scripts文件夹下

options = webdriver.ChromeOptions()

def opt_get_cookie(driver):

    driver.get(r'''https://r.datayes.com/auth/login''')

    change_login_method = '''//*[@id="root"]/div/div[1]/div/div[2]/div/div[2]/div[2]'''
    change_login_method_k = driver.find_element(by=By.XPATH, value=change_login_method)
    change_login_method_k.click()


    user_name = '''huqiuhong@cpic.com.cn'''
    user_pwd = '''b041BYbX'''

    user_input = driver.find_element(by=By.XPATH, value='//input[@type="text"]')
    pwd_input = driver.find_element(by=By.XPATH, value='//input[@type="password"]')

    user_input.send_keys(user_name)
    pwd_input.send_keys(user_pwd)

    login_path = '''//*[@id="rc-tabs-0-panel-password"]/form/div[3]/div/div/span/button'''
    login_btn = driver.find_element(by=By.XPATH, value=login_path)
    login_btn.click()

    cookies = driver.get_cookies()

    with open('test_cath_data\cookies_180.json', 'w') as f:
        f.write(json.dumps(cookies))

opt_get_cookie(driver)
driver.get('https://r.datayes.com/mof/portfolio/overview')  # 此处不要再放登录的网址，可以用未登录的首页

# driver.delete_all_cookies()  # 删除所有cookie信息
# with open('test_cath_data\cookies_180.json', 'r', encoding='utf-8') as f:
#     cookie_list = json.loads(f.read())
# for cookie in cookie_list:
#     driver.add_cookie(cookie)
time.sleep(1)
driver.refresh()

#
target_path = '''//*[@id="root"]/div[1]/div/div[2]/div/div[1]/div[2]/div[2]/button[1]'''

target_btn = driver.find_element(by=By.XPATH, value=target_path)
time.sleep(1)
target_btn.click()
print('1')

time.sleep(1)
# 点开之后调用beautifulSoupw
pp_html = driver.page_source

# target_table_sele = '''/html/body/div[4]/div/div[2]/div/div[2]'''
# target_table = driver.find_element(by=By.CLASS_NAME,value = 'ant-modal-content')

from bs4 import BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'lxml')
# target_table  = soup.find_all('ant-modal-content')
target_soup = soup.select('body > div:nth-child(11) > div > div.ant-modal-wrap.colSettingPop > div > div.ant-modal-content > div.ant-modal-body > div > div.editColSetting_content')
# 获取第一张表
target_table = target_soup[0]

target_table.contents[0]

# 内容部分
info_table = target_table.contents[0].contents[1]
for i in info_table.descendants:
    print(i)