#!/bin/sh
cd /home/ubuntu/gargoyles
/home/ubuntu/env/bin/python2 collect_data.py 168
/home/ubuntu/env/bin/aws s3 sync pricing_data/ s3://gargoyles/pricing_data
rm -r pricing_data/*