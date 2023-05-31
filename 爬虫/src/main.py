import random

import requests
from lxml import etree
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from chaojiying import Chaojiying_Client
import pandas as pd


# 模拟登陆
def init(url: str):
    driver.maximize_window()
    driver.get(url)
    driver.find_element(By.ID, 'loginBtn').click()
    time.sleep(15)


def spyder(key_word, from_page=0, end_page=100):
    res_data = []
    url = f'https://weixin.sogou.com/weixin?query={key_word}&_sug_type_=&sut=167981&lkt=1%2C1679184' \
          f'948505%2C1679184948505&s_from=input&_sug_=n&type=2&sst0=1679185113685&page={str(from_page)}' \
          f'&ie=utf8&w=01019900&dr=1'
    file_path = f'data/data{str(from_page // 15)}.xlsx'
    driver.get(url)

    try:
        for page in range(from_page, end_page):
            url = f'https://weixin.sogou.com/weixin?query={key_word}&_sug_type_=&sut=167981&lkt=1%2C1679184' \
                  f'948505%2C1679184948505&s_from=input&_sug_=n&type=2&sst0=1679185113685&page={str(page)}' \
                  f'&ie=utf8&w=01019900&dr=1'
            # 访问每页的10个网站
            page_source_list = []
            try:
                for i in range(10):
                    if id_code_judge():
                        while True:
                            id_code = identifying_code()
                            print(id_code)
                            id_input = driver.find_element(By.ID, 'seccodeInput')
                            id_input.send_keys(id_code)
                            submit_btn = driver.find_element(By.ID, 'submit')
                            submit_btn.click()
                            time.sleep(5)
                            try:
                                driver.find_element(By.ID, 'sogou_next')
                                break
                            except:
                                continue
                    try:
                        driver.find_element(By.ID, f'sogou_vr_11002601_title_{i}').click()
                        driver.switch_to.window(driver.window_handles[1])
                        time.sleep(3)
                    except:
                        continue
                    try:
                        element = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.ID, 'activity-name')))
                        # element = driver.find_element(By.ID, 'activity-name')
                        print(i)
                        print(page)
                    except:
                        driver.close()
                        driver.switch_to.window(driver.window_handles[0])
                        continue
                    source = driver.page_source
                    page_source_list.append(source)
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    # print(page_source_list)
                # 判断最后一页
                # 如果后面还有页面，那么跳转到后面的页面，end_page+1
                try:
                    next_btn = driver.find_element(By.ID, 'sogou_next')
                    next_btn.click()
                    end_page += 1
                except:
                    break
                # 进行爬虫
                for i in page_source_list:
                    res_data.append(crawl(i))
                    # print(res_data)
                page_source_list.clear()
            except:
                to_xlsx(res_data,file_path)
    except:
        to_xlsx()
    finally:
        to_xlsx(res_data,file_path)


def crawl(data):
    soup = BeautifulSoup(data, 'lxml')
    title = soup.select("#activity-name")[0].get_text().strip()
    date = soup.select('#publish_time')[0].get_text().strip()
    source = soup.select('#js_name')[0].get_text().strip()
    content_list = soup.select('#page-content')[0].get_text().split()
    temp_dict = {len(i): i for i in content_list}
    content = temp_dict[max(temp_dict.keys())].strip()
    return {'Title': title, 'Date': date, 'Source': source, 'Content': content}


def to_xlsx(data_list,file_path):
    data = pd.DataFrame(data_list)
    data.to_excel(file_path, sheet_name='Sheet1', index=False)


def id_code_judge():
    try:
        driver.find_element(By.ID, 'seccodeImage')
        return True
    except:
        return False


def identifying_code():
    vcode_img = driver.find_element(By.ID, 'seccodeImage')
    vcode_img.screenshot('img/identifying_code.png')

    time.sleep(5)
    chaojiying = Chaojiying_Client('leezetu', 'a20031012', '946161')
    im = open('img/identifying_code.png', 'rb').read()
    result = chaojiying.PostPic(im, 1902)['pic_str'].upper()
    time.sleep(5)
    return result


def rand_user_agent():
    user_agent_list = [
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 ",
        "(KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
        "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 "
        "(KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 "
        "(KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 ",
        "(KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 "
        "(KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 "
        "(KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
        "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 ",
        "(KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 "
        "(KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 ",
        "(KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 "
        "(KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 ",
        "(KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 "
        "(KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 ",
        "(KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 "
        "(KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 ",
        "(KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 ",
        "(KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 ",
        "(KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 ",
        "(KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"
    ]
    user_agent = random.choice(user_agent_list)
    return user_agent


def get_proxy():
    with open('data1/proxy.txt', 'r',
              encoding='utf-8') as f:
        proxy_list = f.readlines()
        proxy_list = [i.strip() for i in proxy_list]
    for i in range(len(proxy_list)):
        yield proxy_list[i]


if __name__ == '__main__':
    #driver = webdriver.Firefox()
    # init('https://www.sogou.com')
    # spyder('白菜行情', 3, 100)
    # key_word = '白菜行情'
    all_data = []
    url = 'https://www.sogou.com'
    user_agent = rand_user_agent()
    headers = {'user_agent': user_agent}
    # api_url = 'http://api.proxy.ipidea.io/getBalanceProxyIp?num=100&return_type=txt&lb=1&sb=0&flow=1&regions' \
    #           '=&protocol=http'
    # res = requests.post(api_url, headers=headers, verify=True)
    # proxy = 'https://%s' % res.text
    # print(proxy)
    proxy_iterator = get_proxy()
    key_word = '柑橘价格趋势'

    for i in range(0, 41):
        res = next(proxy_iterator)
        proxy = 'https://%s' % res
        opt = webdriver.FirefoxOptions()
        opt.add_argument('--user-agent=%s' % user_agent)
        opt.add_argument("--proxy-server=%s" % proxy)
        opt.add_argument('--disable-blink-features=AutomationControlled')
        driver = webdriver.Firefox(options=opt)
        init(url)
        spyder(key_word, i * 15 + 1, (i + 1) * 15 + 1)
