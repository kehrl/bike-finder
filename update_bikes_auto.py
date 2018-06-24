#!/usr/bin/env python

from apscheduler.schedulers.blocking import BlockingScheduler
import datetime
import subprocess

sched = BlockingScheduler()

@sched.scheduled_job('interval', seconds=5)
def update_bikes():

    print(datetime.datetime.now())
    subprocess.call(["python","pull_bikes_craigslist.py"])
    subprocess.call(["python","save_postings.py"])
    subprocess.call(["python","predict_bike_types.py"])
    
scheduler = BlockingScheduler()
scheduler.add_job(update_bikes, 'interval', hours=8)
scheduler.start()
 
    
