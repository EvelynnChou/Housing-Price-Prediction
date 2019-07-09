import requests
import json
import lxml
from bs4 import BeautifulSoup
import csv

header = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.90 Safari/537.36'}
BASE_URL = 'https://estate.ltn.com.tw/news/1'


#產生總頁面彙總list
response=requests.get(BASE_URL,headers=header)
soup = BeautifulSoup(response.content,'lxml')
n = int(soup.find(name = 'a', attrs={'class':'p_last'}).get('href')[-3:])

url_list = []
for o in range(1,n+1):  #切換n
    url = 'https://estate.ltn.com.tw/news/{}'.format(o)
    res=requests.get(url,headers=header)
    soup_page=BeautifulSoup(res.content,'lxml')
    sps = soup_page.select('div.page-name a[data-desc="標"]', href=True)
    for href in sps:
        if href['href'].find('video')==-1:
            url_list.append(href['href'])
            
    print(o)


#爬新聞內容
news_list = []
for code in url_list:
    main_url = 'https://estate.ltn.com.tw/{}'.format(code)
    res_url = requests.get(main_url,headers=header)
    news_soup = BeautifulSoup(res_url.text,'html.parser')  #.content,'lxml'

    news_dict = {}
    news_dict['title'] = news_soup.select_one('h1').text.replace('\n','').replace('\r','').replace('<br>','').replace('\r\n','').replace('&nbsp','')
    news_dict['date'] = news_soup.select_one('.time').text[0:10]
    news_dict['content'] = ''.join([tag.text.replace('\n','').replace('\r','').replace('<br>','').replace('\r\n','').replace('&nbsp','') for i,tag in enumerate(news_soup.select('.wordright>p:not(.appE1121)'))])

    news_list.append(news_dict)
    print(code)


#輸出為csv檔    
keys = news_list[0].keys()
with open('crawl_ltn.csv', 'w',encoding='utf8') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(news_list)
