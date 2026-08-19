import sys
import os
import time

for i in range(3):

    print("Sample running: " + str(i))

    # gamma WT 2.5 Gy
    os.system("python .\AMMPERBulk_GAMMAfinal.py d a a 2.5 WT_25k50")

    print("WAITING 60 seconds now")
    time.sleep(60)
    print("START")
