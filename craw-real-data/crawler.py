import requests
from bs4 import BeautifulSoup


class Crawler:
    def __init__(self):
        self.list_url = list()

    def append_path(self, path):
        self.list_url.append(path)

    def crawl(self):
        while len(self.list_url) > 0:
            url = self.list_url.pop()
            print(url)
            response = requests.get(url)
            if response.status_code == 200:
                html = BeautifulSoup(response.content, "html.parser")
                # links = html.findAll('a')
                # for link in links:
                #     list_url.append(link)
                cty = html.findAll('h1')[0]
                print(cty)
            else:
                print('cannot get')


def main():
    path = "https://s.cafef.vn/Lich-su-giao-dich-{name}-1.chn#"
    path = path.format(name="VIC")
    crawl = Crawler()
    crawl.append_path(path)
    crawl.crawl()


if __name__ == "__main__":
    main()
