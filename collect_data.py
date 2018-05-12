import datetime
import pandas as pd
import requests
import time
import os
import pdb
import sys 

coins = ['BTC','ETH', 'BCH','LTC', 'EOS', 'NEO','TRX', 'XRP', 'ADA', 'XLM','MIOTA', 'DASH','XMR','BCN','ETC','BNB','ICX','QTUM']

def hourly_price_historical(symbol, comparison_symbol, limit=168, aggregate=1, exchange='binance'):
    url = 'https://min-api.cryptocompare.com/data/histohour?fsym={}&tsym={}&limit={}&aggregate={}'\
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
   
    if exchange:
        url += '&exchange={}'.format(exchange)

    page = requests.get(url)

    data = page.json()['Data']
    df = pd.DataFrame(data)

    df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df['time']]

    return df


def get_save_data(limit):

    dfs = []

    for coin in coins:
        try:
            data = hourly_price_historical(coin,'USD', limit)

            # Save to individual coins data
            if not (os.path.exists('pricing_data/each_coin/%s'%coin)):
                os.makedirs('pricing_data/each_coin/%s/'%coin)

            data.to_csv('pricing_data/each_coin/%s/initial_bulk_%s.csv'%(coin,time.strftime("%Y-%m-%d")))

            # Prefix col name with coin
            data.columns = [(coin + '_'+col) if col !="timestamp" else col for col in data.columns  ]
            
            # Append to all dfs array
            dfs.append(data)
        except:
            print('Error with coin %s'%coin)
    
    # Merge all dfs on name col
    df_final = reduce(lambda left,right: pd.merge(left,right,on='timestamp'), dfs)

    # Save to csv
    if not (os.path.exists('pricing_data/all_coins')):
        os.makedirs('pricing_data/all_coins')
    
    df_final.to_csv('pricing_data/all_coins/initial_bulk_%s.csv'%(time.strftime("%Y-%m-%d")))


if __name__=="__main__":

    limit = sys.argv[1]

    get_save_data(limit)