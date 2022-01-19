import json


class Stock:
    def __int__(self, company="", short_name="", date=None, close_price=0, open_price=0):
        self.company = company
        self.short_name = short_name
        self.date = date
        self.close_price = close_price
        self.open_price = open_price

    def to_json(self):
        record = dict()
        record['company'] = self.company
        record['short name'] = self.short_name
        record['date'] = self.date
        record['close'] = self.close_price
        record['open'] = self.open_price
        js = json.dumps(record)
        return js


class StockDataSet:
    def __init__(self, output_path="../dataset/", company="", year=None, url_collect=""):
        self.dataset = []
        self.output_path = output_path
        self.url_collect = url_collect
        self.company = company
        self.year = year

    def write_to_csv(self):
        js = self.csv_header() + '\n'
        for data in self.dataset:
            js += data.to_json() + "\n"
        with self.output_path as f:
            f.write(js)

    def csv_header(self):
        header = ""
        header += "\n url collection: " + self.url_collect
        header += "\n company: " + self.company
        header += "\n year: " + self.year
        header += "\n company, short name, date, close, open"
        return header
