import multiprocessing
import subprocess
import sys
import time
import os
import psutil
import tkinter as tk
from tkinter import ttk

def Bar():
    subprocess.run(["python", "ProgressBar.py"])



def ModelMaker():
    pid = os.getpid()
    subprocess.run(["python", "SketchToModel.py"])
    sys.exit()

#def Update(pid):
 #   while not psutil.pid_exists(pid)


p1 = multiprocessing.Process(target=Bar)
p2 = multiprocessing.Process(target=ModelMaker)

if __name__ == '__main__':
    p1.start()
    p2.start()
    p1.join()
    p2.join()


#pid = p2.pid
finish = time.perf_counter()
print("Finished running after: ",finish)

#while psutil.pid_exists(pid):
 #   print(pid)
