import requests
import parsel
url="https://sh.lianjia.com/ershoufang/rs/"
headers={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.9 Safari/537.36'
}
response=requests.get(url=url,headers=headers)
#print(response.text)
selector=parsel.Selector(response.text)
href=selector.css('.sellListContent Li .title a::attr(href)').getall()
#print(href)
for link in href:
    response1=requests.get(url=link,headers=headers)
    print(response1.text)
    print(response1.content)
    #print(link)
    break