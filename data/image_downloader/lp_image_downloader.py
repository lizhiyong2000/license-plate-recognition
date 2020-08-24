# -*- coding: utf-8 -*-
# author: Li Zhiyong
# Email: lizhiyong2000@gmail.com

import argparse
import sys
from concurrent import futures

from data.image_downloader import crawler, downloader


def main(argv):
    parser = argparse.ArgumentParser(description="Image Downloader")
    parser.add_argument("--keywords", type=str, default="车牌",
                        help='Keywords to search. ("in quotes")')
    parser.add_argument("--engine", "-e", type=str, default="Baidu",
                        help="Image search engine.", choices=["Google", "Bing", "Baidu"])
    parser.add_argument("--driver", "-d", type=str, default="chrome_headless",
                        help="Image search engine.", choices=["chrome_headless", "chrome", "phantomjs"])
    parser.add_argument("--max-number", "-n", type=int, default=100,
                        help="Max number of images download for the keywords.")
    parser.add_argument("--num-threads", "-j", type=int, default=10,
                        help="Number of threads to concurrently download images.")
    parser.add_argument("--timeout", "-t", type=int, default=20,
                        help="Seconds to timeout when download an image.")
    parser.add_argument("--output", "-o", type=str, default="../../test_pic/train_images",
                        help="Output directory to save downloaded images.")
    parser.add_argument("--safe-mode", "-S", action="store_true", default=False,
                        help="Turn on safe search mode. (Only effective in Google)")
    parser.add_argument("--face-only", "-F", action="store_true", default=False,
                        help="Only search for ")
    parser.add_argument("--proxy_http", "-ph", type=str, default=None,
                        help="Set http proxy (e.g. 192.168.0.2:8080)")
    parser.add_argument("--proxy_socks5", "-ps", type=str, default=None,
                        help="Set socks5 proxy (e.g. 192.168.0.2:1080)")

    args = parser.parse_args(args=argv)

    proxy_type = None
    proxy = None
    if args.proxy_http is not None:
        proxy_type = "http"
        proxy = args.proxy_http
    elif args.proxy_socks5 is not None:
        proxy_type = "socks5"
        proxy = args.proxy_socks5

    max_number = 1000
    batch_size = 60
    current_batch = 0
    current_number = 0
    uncompleted = 0

    with futures.ThreadPoolExecutor(max_workers=5) as executor:
        crawled_urls = list()
        future_list = list()
        while current_number + uncompleted < max_number:
            future_list.append(executor.submit(crawler.crawl_image_urls, args.keywords, batch_no=current_batch, batch_size=batch_size))
            current_batch += 1
            uncompleted += batch_size

        for future in futures.as_completed(future_list):
            if future.exception() is None:
                crawled_urls += future.result()
            else:
                print(future.exception())

        count = downloader.download_images(image_urls=crawled_urls, dst_dir=args.output,
                                   concurrency=args.num_threads, timeout=args.timeout,
                                   proxy_type=proxy_type, proxy=proxy,
                                   file_prefix=args.engine)

        print("{0} downloaded".format(count))

    print("Finished.")


if __name__ == '__main__':
    main(sys.argv[1:])