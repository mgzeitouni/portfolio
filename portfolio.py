from binance.client import Client
from credentials import *
client = Client(api_key, api_secret)

#orders = client.get_all_orders(symbol='EOS/ETH', limit=10)
#print(orders)

balances = client.get_account()['balances']

positive_balances = [x for x in balances if float(x['free'])>0]

print(positive_balances)