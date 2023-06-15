import time
from tkinter import *
from tkinter import ttk
import psutil

root = Tk()
root.title("Imagin3D")
root.geometry("1920x1080")
root.attributes('-fullscreen',True)

bg = PhotoImage(file = "P-1_1686128525452-1000.png")

label1 = Label(root, image=bg)
label1.place(x=0,y=0)

my_progress = ttk.Progressbar(root, orient=HORIZONTAL, length=300, mode='indeterminate')
my_progress.pack(pady=(700, 20))
my_progress.start(10)

if ("SketchToModel.py" in (i.name() for i in psutil.process_iter()) == True):
    print("Running")

exit_button = Button(root, text="Exit", command=root.destroy)
exit_button.pack(pady=20)

root.mainloop()