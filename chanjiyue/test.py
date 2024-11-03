import requests
from bs4 import BeautifulSoup

url='https://sh.lianjia.com/ershoufang/'
headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.9 Safari/537.36'
}
r=requests.get(url,headers=headers)
print(r.text)
