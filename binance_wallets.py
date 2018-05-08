from binance.client import Client
from credentials import *
client = Client(binance_api_key, binance_api_secret)
import json
#orders = client.get_all_orders(symbol='EOS/ETH', limit=10)
#print(orders)

def get_binance_balances():
    balances = client.get_account()['balances']

    positive_balances = [{x['asset']:float(x['free'])+float(x['locked'])} for x in balances if float(x['free'])>0]

    return json.dumps(positive_balances)