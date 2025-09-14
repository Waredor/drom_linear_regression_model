import os
import requests
import json
from bs4 import BeautifulSoup


class Parser:
    """
    Класс Parser осуществляет парсинг веб-страниц, передаваемых в словаре base_url_dict.
    Parameters:
        req_headers (dict): заголовки для http запроса на сервер
        base_url_dict (dict):
    """
    def __init__(self, req_headers: dict, base_url_dict: dict, class_name: str, output_filepath: str) -> None:
        self.base_url_dict = base_url_dict
        self.class_name = class_name
        self.req_headers = req_headers
        self.output_filepath = output_filepath

    def parse_by_url(self, url: str, tag: str) -> tuple:
        src = requests.get(url, headers=self.req_headers)
        htmldata = BeautifulSoup(src.text, 'lxml')
        data = htmldata.find_all(str(tag), class_=self.class_name)
        return src, data

    def collect_elements(self) -> None:
        for key, value in self.base_url_dict.items():
            src = self.parse_by_url(url=value, tag='div')[0]
            parsed_links = self.parse_by_url(url=value, tag='div')[-1]
            listoflinks = []

            i = 1
            for dt in parsed_links:
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

                listoflinks.append({f'car {i}': [{'price': price}, {'odo': odo}, {'year': year}]})
                i += 1

            htmldata = BeautifulSoup(src.text, 'lxml')
            next_page_relative_url = htmldata.find('a', class_='_1j1e08n0 _1j1e08n5')['href']

            while next_page_relative_url is not None:
                src = self.parse_by_url(url=next_page_relative_url, tag='div')[0]
                htmldata = BeautifulSoup(src.text, 'lxml')
                parsed_links = self.parse_by_url(next_page_relative_url, tag='div')[-1]

                for dt in parsed_links:
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

            filename = f"{key}.json"
            json_filepath = os.path.join(self.output_filepath, filename)
            with open(json_filepath, 'w', encoding='utf-8') as file:
                json.dump(listoflinks, file, ensure_ascii=False)