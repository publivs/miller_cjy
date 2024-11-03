import requests
from bs4 import BeautifulSoup

url='https://u.ccb.com/lib/reader/book/ePub/getEPub'
headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.9 Safari/537.36'
}
cookie = '''JSESSIONID=E5B50844E11D7796E608AC3C951B1E05; ELEARNING_00999=x1vx4r3c2boz2ghulpc24ci1; ELEARNING_00003=StudyMenuGroup; route=3dc5695ec4abb875f575e7a95e5e7817; isNewIndex=true; trkslacu=0c776eb2-4f4c-4e21-b4ef-5c006c1f61ab; ELEARNING_00095=x1vx4r3c2boz2ghulpc24ci1; ELEARNING_00018=evUKJsUrEccRmsaQ+tFvXOis190LAFkdIwPpKPGsjA7WeeQS5huU9MMwVSBAO9uX; ELEARNING_00024=ticket-1-aeRNnMND4yX9fevHdKdVicLxi3FdQzxTZG1; ELEARNING_00017=8f221f83-7521-453a-95f7-662c9d112795; ELEARNING_00002=chanjiyue.sh; ELEARNING_00096=chrome113.0; XXTOWN_COOKIE_00018=5e4d451e-7f4c-4dd5-9577-90be05189a1d; SassType=0; loginType=HnPwd; context=/lib; dlib_jsessionid=0692d397877aaa1a94c5047ac28ab550'''
payload = {'key1': 'value1', 'key2': 'value2'}
response = requests.post(url, data=payload)
print(response.text)
