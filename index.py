from flask import Flask, render_template, request, jsonify
import os
import json
from binance_wallets import *
from coinbase_wallets import *

app = Flask(__name__, static_url_path='')


port = int(os.getenv('PORT', 8000))

@app.route('/')
def root():
  return 'hey world'

@app.route('/binance-balances', methods=['POST'])
def balances():
    if ('password' in request.headers and request.headers['password'] =='super-secret-crypto-password'):
        return get_binance_balances()
    else:
        return "Incorrect password"

@app.route('/coinbase-balances', methods=['POST'])
def coinbase_balances():
    if ('password' in request.headers and request.headers['password'] =='super-secret-crypto-password'):
        return get_coinbase_balances()
    else:
        return "Incorrect password"
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
