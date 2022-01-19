import yfinance as yf


def download_data(tickers=None, start='2021-01-01', end='2021-11-16', intervals='1m', path="../dataset/"):
    """
        Download stock price data and store to path
        ...
        Arguments:
            tickers: str
                list of company tickers (default None)
            start: str
                crawl data from start time (default "2021-01-01")
            end: str
                crawl data to end time (default "2021-11-16")
            intervals: str
                intervals of data row (default "1m")
            path: str
                path to save csv file (default "../dataset/"
    """
    for ticker in tickers:
        print('downloading', ticker)
        gg = yf.download(ticker, start=start, end=end, intervals=intervals)
        gg.to_csv(path + ticker + ".csv")


if __name__ == '__main__':
    tickers = ['FB', 'MSFT', 'AAPL', 'SPY', 'ELEK', 'GOEV', 'LCID', 'QCOM', 'NVDA', 'GOOG']
    download_data(tickers, "2016-01-01", '2021-01-01', '1wk')
