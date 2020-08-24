""" Crawl image urls from image search engine. """
# -*- coding: utf-8 -*-
# author: Yabin Zheng
# Email: sczhengyabin@hotmail.com



import re
import time
import sys
import os
import json
import codecs
import shutil

from urllib.parse import unquote, quote
from selenium import webdriver
from selenium.webdriver import DesiredCapabilities
import requests
from concurrent import futures

if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
else:
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

dcap = dict(DesiredCapabilities.PHANTOMJS)
dcap["phantomjs.page.settings.userAgent"] = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.100"
)


def my_print(msg, quiet=False):
    if not quiet:
        print(msg)

def baidu_gen_query_url(keywords, face_only=False, safe_mode=False):
    base_url = "https://image.baidu.com/search/index?tn=baiduimage"
    keywords_str = "&word=" + quote(keywords)
    query_url = base_url + keywords_str
    if face_only is True:
        query_url += "&face=1"
    return query_url


def baidu_image_url_from_webpage(driver):
    time.sleep(10)
    image_elements = driver.find_elements_by_class_name("imgitem")
    image_urls = list()

    for image_element in image_elements:
        image_url = image_element.get_attribute("data-objurl")
        image_urls.append(image_url)
    return image_urls


def baidu_get_image_url_using_api(keywords, batch_no=0, batch_size=60, face_only=False,
                                  proxy=None, proxy_type=None):
    def decode_url(url):
        in_table = '0123456789abcdefghijklmnopqrstuvw'
        out_table = '7dgjmoru140852vsnkheb963wtqplifca'
        translate_table = str.maketrans(in_table, out_table)
        mapping = {'_z2C$q': ':', '_z&e3B': '.', 'AzdH3F': '/'}
        for k, v in mapping.items():
            url = url.replace(k, v)
        return url.translate(translate_table)

    base_url = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&lm=7&fp=result&ie=utf-8&oe=utf-8&st=-1"
    keywords_str = "&word={}&queryWord={}".format(
        quote(keywords), quote(keywords))
    query_url = base_url + keywords_str
    query_url += "&face={}".format(1 if face_only else 0)

    # init_url = query_url + "&pn=0&rn=30"

    proxies = None
    if proxy and proxy_type:
        proxies = {"http": "{}://{}".format(proxy_type, proxy),
                   "https": "{}://{}".format(proxy_type, proxy)}

    # res = requests.get(init_url, proxies=proxies)
    # init_json = json.loads(res.text.replace(r"\'", ""), encoding='utf-8', strict=False)
    # total_num = init_json['listNum']

    # target_num = min(max_number, total_num)
    # crawl_num = min(target_num * 2, total_num)

    crawled_urls = list()
    # batch_size = 30

    # with futures.ThreadPoolExecutor(max_workers=5) as executor:
    #     future_list = list()

        # def process_batch(batch_no, batch_size):
    image_urls = list()
    url = query_url + \
        "&pn={}&rn={}".format(batch_no * batch_size, batch_size)

    my_print("Query URL: " + url, quiet=False)
    try_time = 0
    while True:
        try:
            response = requests.get(url, proxies=proxies)
            break
        except Exception as e:
            try_time += 1
            if try_time > 3:
                print(e)
                return image_urls
    response.encoding = 'utf-8'
    res_json = json.loads(response.text.replace(r"\'", ""), encoding='utf-8', strict=False)
    for data in res_json['data']:
        if 'objURL' in data.keys():
            image_urls.append(decode_url(data['objURL']))
        elif 'replaceUrl' in data.keys() and len(data['replaceUrl']) == 2:
            image_urls.append(data['replaceUrl'][1]['ObjURL'])

    return image_urls

    #     for i in range(0, int((crawl_num + batch_size - 1) / batch_size)):
    #         future_list.append(executor.submit(process_batch, i, batch_size))
    #     for future in futures.as_completed(future_list):
    #         if future.exception() is None:
    #             crawled_urls += future.result()
    #         else:
    #             print(future.exception())
    #
    # return crawled_urls[:min(len(crawled_urls), target_num)]


def crawl_image_urls(keywords, engine="baidu", batch_no=0, batch_size=60,
                     face_only=False, safe_mode=False, proxy=None, 
                     proxy_type="http", quiet=False, browser="chrome"):
    """
    Scrape image urls of keywords from Google Image Search
    :param keywords: keywords you want to search
    :param engine: search engine used to search images
    :param max_number: limit the max number of image urls the function output, equal or less than 0 for unlimited
    :param face_only: image type set to face only, provided by Google
    :param safe_mode: switch for safe mode of Google Search
    :param proxy: proxy address, example: socks5 127.0.0.1:1080
    :param proxy_type: socks5, http
    :param browser: browser to use when crawl image urls from Google & Bing 
    :return: list of scraped image urls
    """

    my_print("\nScraping From {0} Image Search ...\n".format(engine), quiet)
    my_print("Keywords:  " + keywords, quiet)
    # if max_number <= 0:
    #     my_print("Number:  No limit", quiet)
    #     max_number = 10000
    # else:
    #     my_print("Number:  {}".format(max_number), quiet)
    my_print("Face Only:  {}".format(str(face_only)), quiet)
    my_print("Safe Mode:  {}".format(str(safe_mode)), quiet)


    query_url = baidu_gen_query_url(keywords, face_only, safe_mode)

    # my_print("Query URL:  " + query_url, quiet)

    # browser = str.lower(browser)
    # if "chrome" in browser:
    #     chrome_path = shutil.which("chromedriver")
    #     chrome_path = "./bin/chromedriver" if chrome_path is None else chrome_path
    #     chrome_options = webdriver.ChromeOptions()
    #     if "headless" in browser:
    #         chrome_options.add_argument("headless")
    #     if proxy is not None and proxy_type is not None:
    #         chrome_options.add_argument("--proxy-server={}://{}".format(proxy_type, proxy))
    #     driver = webdriver.Chrome(chrome_path, chrome_options=chrome_options)
    # else:
    #     phantomjs_path = shutil.which("phantomjs")
    #     phantomjs_path = "./bin/phantomjs" if phantomjs_path is None else phantomjs_path
    #     phantomjs_args = []
    #     if proxy is not None and proxy_type is not None:
    #         phantomjs_args += [
    #             "--proxy=" + proxy,
    #             "--proxy-type=" + proxy_type,
    #         ]
    #     driver = webdriver.PhantomJS(executable_path=phantomjs_path,
    #                                  service_args=phantomjs_args, desired_capabilities=dcap)


    # driver.set_window_size(10000, 7500)
    # driver.get(query_url)
    # image_urls = baidu_image_url_from_webpage(driver)
    image_urls = baidu_get_image_url_using_api(keywords, batch_no=batch_no, batch_size=batch_size, face_only=face_only,
                                               proxy=proxy, proxy_type=proxy_type)

    # driver.close()

    # if max_number > len(image_urls):
    #     output_num = len(image_urls)
    # else:
    #     output_num = max_number

    my_print("\n== batch:{0} with size {1} crawled {2} images urls will be used.\n".format(
        batch_no, batch_size, len(image_urls)), quiet)

    return image_urls
