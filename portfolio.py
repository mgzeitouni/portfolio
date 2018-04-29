from binance.client import Client

api_key = 'ShEUhsQ3QhIi0uUNngOB8Qhl86dMUNgyZWIaZ4FPzF9OvXcNY8MfuxndmB6PpOE5'
api_secret = 'cUvRL7x9gtRWqEodnBmRYd8ILcwT5IcIS3cz9pwbTfan14nOTdSIwzlW95YgCCWb'
client = Client(api_key, api_secret)

#orders = client.get_all_orders(symbol='EOS/ETH', limit=10)
#print(orders)

balances = client.get_account()['balances']

positive_balances = [x for x in balances if float(x['free'])>0]

print(positive_balances)