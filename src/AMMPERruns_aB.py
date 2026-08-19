import sys
import os

for i in range(2):

    print("Sample running: " + str(i))

    # WT Basic 2.5 Gy
    os.system("python .\AMMPERBulk_aB.py a a a 0 WT_Basic_0")
    os.system("python .\AMMPERBulk_aB.py a a a 2.5 WT_Basic_25")
    os.system("python .\AMMPERBulk_aB.py a a a 30 WT_Basic_300")
    
    # 