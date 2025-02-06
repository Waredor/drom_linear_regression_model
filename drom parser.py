import requests
import json
from bs4 import BeautifulSoup

def parse_by_url(baseurl, ourheaders, classname, tag):
   src = requests.get(str(baseurl), headers=ourheaders)
   htmldata = BeautifulSoup(src.text, 'lxml')
   data = htmldata.find_all(str(tag), class_=str(classname))
   return data


#задаем заголовки для htpp запроса на сервер
st_accept = "text/html"
st_useragent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
headers = {
   "Accept": st_accept,
   "User-Agent": st_useragent
}

#задаем url сайта
oururl = "https://habarovsk.drom.ru/toyota/prius/generation3/restyling1/"


#задаем параметры поиска на первой странице сайта
linksdata = parse_by_url(oururl, headers, classname='css-1f68fiz ea1vuk60', tag='div')
listoflinks = []
i = 1
for dt in linksdata:
   price = dt.find('span', 'css-46itwz e162wx9x0').text
   price = str(price).split()
   price.pop(-1)
   price = int(''.join(price))
   odo_data = dt.find_all('span', 'css-1l9tp44 e162wx9x0')
   odo = odo_data[-1].text
   odo = str(odo).split()
   odo.pop(-1)
   odo = int(''.join(odo))
   year = dt.find('h3', 'css-16kqa8y efwtv890')
   if year is None:
      pass
   else:
      year = year.text.split(',')
      year = int(year[1])
   listoflinks.append({f'car {i}':[{'price': price},{'odo': odo}, {'year': year}]})
   i += 1


#ищем якорь на следующие страницы и реализуем логику обхода этих страниц
src = requests.get(oururl, headers=headers)
htmldata = BeautifulSoup(src.text, 'lxml')
next_page_relative_url = htmldata.find('a', class_='_1j1e08n0 _1j1e08n5')['href']

while next_page_relative_url is not None:
   src = requests.get(next_page_relative_url, headers=headers)
   htmldata = BeautifulSoup(src.text, 'lxml')
   linksdata = parse_by_url(next_page_relative_url, headers, classname='css-1f68fiz ea1vuk60', tag='div')
   for dt in linksdata:
      price = dt.find('span', 'css-46itwz e162wx9x0').text
      price = str(price).split()
      price.pop(-1)
      price = int(''.join(price))
      odo_data = dt.find_all('span', 'css-1l9tp44 e162wx9x0')
      odo = odo_data[-1].text
      odo = str(odo).split()
      year = dt.find('h3', 'css-16kqa8y efwtv890')
      if year is None:
         pass
      else:
         year = year.text.split(',')
         year = int(year[1])
      if odo[0] == '(≈':
         odo.pop(0)
      odo.pop(-1)
      odo = ''.join(odo)
      if len(odo) < 5:
         continue
      listoflinks.append({f'car {i}': [{'price': price}, {'odo': odo}, {'year': year}]})
      i += 1

   if htmldata.find('a', class_='_1j1e08n0 _1j1e08n5') is not None:
      next_page_relative_url = htmldata.find('a', class_='_1j1e08n0 _1j1e08n5')['href']
   else:
      next_page_relative_url = None



with open('prius_drom_khv.json', 'w') as file:
   json.dump(listoflinks, file, ensure_ascii=False)

