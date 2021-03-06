#!/usr/bin/env python
'''
Script to automatically update SQL database with new, LABELED bike postings and delete old ones.

Laura Kehrl 
'''

from apscheduler.schedulers.blocking import BlockingScheduler
import datetime
import subprocess
import numpy as np

sched = BlockingScheduler()

def update_bikes():

    start_time = datetime.datetime.now()
    print("Started at", start_time)
    
    subprocess.call(["python","pull_bikes_craigslist.py"])
    subprocess.call(["python","save_postings.py"])
    subprocess.call(["python","predict_bike_types.py"])
    
    delta_time = datetime.datetime.now() - start_time
    print("Elapsed hours", np.round(delta_time.seconds/(60*60),1))

update_bikes()
scheduler = BlockingScheduler()
scheduler.add_job(update_bikes, 'interval', hours=3)
scheduler.start()
