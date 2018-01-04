from __future__ import unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas.tools.plotting as plotting
import datetime as dt

NUMBER = ['2914',
          '4503',
          '6758',
         ]
start = '1998-01-01'
end = '2017-12-31'

def get_quote_yahoojp(code, start=None, end=None, interval='d'):
    base = 'http://info.finance.yahoo.co.jp/history/?code={0}.T&{1}&{2}&tm={3}&p={4}'
    #start, end = web._sanitize_dates(start, end)
    start = pd.to_datetime(start)
    if end == None:
        end = pd.to_datetime(pd.datetime.now())
    else :
        end = pd.to_datetime(end)
    start = 'sy={0}&sm={1}&sd={2}'.format(start.year, start.month, start.day)
    end = 'ey={0}&em={1}&ed={2}'.format(end.year, end.month, end.day)
    p = 1
    results = []

    if interval not in ['d', 'w', 'm', 'v']:
        raise ValueError("Invalid interval: valid values are 'd', 'w', 'm' and 'v'")

    while True:
        url = base.format(code, start, end, interval, p)
        tables = pd.read_html(url, header=0)
        if len(tables) < 2 or len(tables[1]) == 0:
            break
        results.append(tables[1])
        p += 1
    result = pd.concat(results, ignore_index=True)

    result.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    if interval == 'm':
        result['Date'] = pd.to_datetime(result['Date'], format='%Y年%m月')
    else:
        result['Date'] = pd.to_datetime(result['Date'], format='%Y年%m月%d日')
    result = result.set_index('Date')
    result = result.sort_index()
    return result

if __name__ == '__main__':
    for company in NUMBER:
        df = get_quote_yahoojp(company, start=start, end=end)
        df.to_csv(company+".csv")
        print(company, "OK.")