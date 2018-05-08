from coinbase.wallet.client import Client
from credentials import *
client = Client(coinbase_api_key, coinbase_api_secret)
import json

def get_coinbase_balances():
    accounts = client.get_accounts()

    positive_balances = [{x['balance']['currency']:float(x['balance']['amount'])} for x in accounts['data'] if float(x['balance']['amount'])>0]

    return json.dumps(positive_balances)

