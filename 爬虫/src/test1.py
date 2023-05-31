import requests

def get_proxy():
    with open('/data1/proxy.txt', 'r', encoding='utf-8') as f:
        proxy_list = f.readlines()
        proxy_list = [i.strip() for i in proxy_list]
    for i in range(len(proxy_list)):
        yield proxy_list[i]
a = get_proxy()
print(next(a))
print(next(a))