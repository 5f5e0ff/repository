import numpy as np
import pandas as pd
import pandas_datareader.data as web

INDEIESaa = ['^AORD',
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
INDEIESaa = [#'^AORD',
           #'^N225',
           #'^HSI',
           #'^GDAXI',
           #'^FTSE',
           #'^NYA',
           #'^DJI',
           '^GSPC'
          ]
INDEIES = [#'^FCHI',
           #'^AEX',
           #'^IXIC',
           #'JPY=X',
           'EURUSD=X',
           'GBPUSD=X',
          ]
start = '1998-01-01'
end = '2017-12-31'

def TF_ohlc(df, tf):
    x = df.resample(tf).ohlc()
    O = x['Open']['open']
    H = x['High']['high']
    L = x['Low']['low']
    C = x['Close']['close']
    ret = pd.DataFrame({'Open': O, 'High': H, 'Low': L, 'Close': C},
                       columns=['Open','High','Low','Close'])
    return ret.dropna()

if __name__ == "__main__":
    for stock in INDEIES:
        df = web.DataReader(stock, 'yahoo', start=start, end=end)
        df_w = TF_ohlc(df, 'W')
        df_w.to_csv(stock+".csv")
        print(stock, "OK.")