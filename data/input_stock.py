import numpy as np
import pandas as pd
import pandas_datareader.data as web

INDEIES = ['^AORD',
           '^STOXX50E',
           '^FCHI',
           '^GDAXI',
           '^HSI',
           'FTSEMIB.MI',
           '^N225',
           '^AEX',
           'WIG.PA',
           '^IBEX',
           '^FTSE',
           '^IXIC',
           '^DJI',
           '^GSPC',
           'JPY=X',
           'EURJPY=X',
           'GBPJPY=X',
          ]
start = '2005-01-01'
end = None

if __name__ == "__main__":
    for stock in INDEIES:
        df = web.DataReader(stock, 'yahoo', start=start, end=end)
        df.to_csv(stock+".csv")
        print(stock, "OK.")