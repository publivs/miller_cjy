import requests
from bs4 import BeautifulSoup
import chardet
from selenium import webdriver
url = 'http://www.pbc.gov.cn/rmyh/105208/index.html'
driver = webdriver.Chrome()  # 使用Chrome浏览器
driver.get(url)  # 打开网页
html = driver.page_source  # 获取网页源代码
driver.quit()  # 关闭浏览器
print(html)

# url = 'http://www.pbc.gov.cn/goutongjiaoliu/113456/113469/11040/index.html'
# url ='http://www.pbc.gov.cn/rmyh/105208/index.html'
# response = requests.get(url)
# response.encoding = 'utf-8'  # 修改编码方式
# soup = BeautifulSoup(response.text, 'html.parser')
# data = []
# for link in soup.find_all('a', href=True):
#     if '货币政策例会纪要' in link.text:
#         href = link['href']
#         title = link.text.strip()
#         title = title.encode('ISO-8859-1').decode(chardet.detect(title)['encoding']) # 转换字符集
#         data.append((title, href))
# print(data)