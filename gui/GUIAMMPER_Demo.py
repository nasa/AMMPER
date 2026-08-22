from sys import argv
from tkinter import *
import random
import glob
import os
from PIL import Image, ImageTk
import time
import tkinter as tk

import tkinter as tk

#########33 attempt 1: Try to  create a global path, then use command functions to edit this global path, then show global path movies

class SampleApp2(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        container = tk.Frame(self)
        container.grid()
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        F = PLG
        page_name = F.__name__
        frame = F(parent=container, controller=self) # <----
        self.frames[page_name] = frame # <--- Contains all Start Pages objects, access by dictionary with pages name
        frame.grid(row=0, column=0, sticky="nsew")

        self.config(bg="light cyan")
        self.title('AMMPER')
        self.show_frame("PLG")

    def show_frame(self, page_name):
        for frame in self.frames.values():
            frame.grid_remove()
        frame = self.frames[page_name]
        frame.grid()

class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        container = tk.Frame(self)
        container.grid()
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        F = StartPage
        page_name = F.__name__
        frame = F(parent=container, controller=self) # <----
        self.frames[page_name] = frame # <--- Contains all Start Pages objects, access by dictionary with pages name
        frame.grid(row=0, column=0, sticky="nsew")
######################################################33
        #
        # image1 = Image.open(r'C:\Users\dmpalaci\Documents\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\BPS.png')
        # image1 = image1.resize((215, 100), Image.ANTIALIAS)
        # test = ImageTk.PhotoImage(image1)
        #
        # label1 = tk.Label(image=test)
        # label1.image = test
        # label1.grid(row = 1, column = 0)


        ####################################33
        # colors and main gidgets
        button1 = tk.Button(self, text="Start AMMPER", command=self.close)
        button1.config(bg="floral white")
        button1.grid()
        self.config(bg="light cyan")
        self.title('AMMPER')
        self.show_frame("StartPage")

    def show_frame(self, page_name):
        for frame in self.frames.values():
            frame.grid_remove()
        frame = self.frames[page_name]
        frame.grid()

    def close(self):
        Tk().destroy()
        self.destroy()

class StartPage(tk.Frame):
    # MASTER PATH locating MOVIES for GUI or AMMPER RESULTS FILE >>> <<< >>> <<<
    path = r"C:\Users\dmpalaci\Documents\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\GUI_MOVIES"
    pathk = ''
    pathkk = ''

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller

        label = tk.Label(self, text="Select Inputs", font=("Verdana Bold", 12),
            bg = 'light cyan')

        label.grid(row = 0, column = 2)
        ########### COLORS << >>>
        self.config(bg="light cyan")

        self.selected_value = tk.IntVar()
        self.selected_value2 = tk.IntVar()
        self.selected_value3 = tk.IntVar()

        self.create_widgets()



    def RadiationType(self):

        self.choice = self.selected_value.get()
        self.RadType = self.selected_value.get()
        self.output = "150 MeV Proton"

        if self.RadType == 1:
            print("150 MeV Proton")
            self.output = "150 MeV Proton"

        elif self.RadType == 2:
            print("GCRSim")
            self.output = "GCRSim"

        elif self.RadType == 3:
            print("Deep Space")
            self.output = "Deep Space"

        # self.path = r"C:\Users\dmpalaci\Documents\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\GUI_MOVIES"

        self.pathk = self.pathk

        a = self.output
        self.pathkk = os.path.join(self.pathk, a)

        return self.output

    def RadiationLevel(self):
        # global output2
        self.choice2 = self.selected_value2.get()
        self.RadLev = self.selected_value2.get()
        self.output2 = "0 Gy"

        if self.RadLev == 1:
            print("0 Gy")
            self.output2 = "0 Gy"

        elif self.RadLev == 2:
            print("2.5 Gy")
            self.output2 = "2.5 Gy"

        elif self.RadLev == 3:
            print("5 Gy")
            self.output2 = "5 Gy"

        elif self.RadLev == 4:
            print("10 Gy")
            self.output2 = "10 Gy"

        elif self.RadLev == 5:
            print("20 Gy")
            self.output2 = "20 Gy"

        elif self.RadLev == 6:
            print("30 Gy")
            self.output2 = "30 Gy"

        a = self.output2

        self.pathkk = self.pathkk
        
        self.pathkkk = os.path.join(self.pathkk, a)

        self.pathkkk = os.path.join(self.pathkkk, "*.png")
        ###################################3
        print(self.pathkkk)

        return self.pathkkk

    def CellTypee(self):

        self.choice3 = self.selected_value3.get()
        self.CellT = self.selected_value3.get()
        # default value
        self.output3 = "WildType"

        if self.CellT == 1:
            print("Wild Type")
            self.output3 = "WildType"

        elif self.CellT == 2:
            print("Rad51")
            self.output3 = "Rad51"

        # self.path =
        a = self.output3
        self.pathk = os.path.join(self.path, a)

        return self.output3

    def create_widgets(self):

        color_background = 'light cyan'

        self.label0 = tk.Label(
            self,
            text="Choose the Radition Type:",
            font=("Verdana Bold", 11),
            bg = color_background)

        self.rb1 = tk.Radiobutton(
            self,
            text="150 MeV Proton",
            padx=5,
            pady=5,
            variable=self.selected_value,
            command=self.RadiationType,
            value=1,
            font=("Verdana", 11),
            bg = color_background)

        self.rb2 = tk.Radiobutton(
            self,
            text="GCRSim",
            padx=5,
            pady=5,
            variable=self.selected_value,
            command=self.RadiationType,
            value=2,
            font=("Verdana", 11),
            bg = color_background)

        self.rb3 = tk.Radiobutton(
            self,
            text="Deep Space",
            padx=5,
            pady=5,
            variable=self.selected_value,
            command=self.RadiationType,
            value=3,
            font=("Verdana", 11),
            bg = color_background)

        self.label1 = tk.Label(
            self,
            text="Choose Radition Level (Gy):",
            font=("Verdana Bold", 11),
            bg = color_background)

        self.rrb1 = tk.Radiobutton(
            self,
            text="0",
            padx=5,
            pady=5,
            variable=self.selected_value2,
            command=self.RadiationLevel,
            value=1,
            font=("Verdana", 11),
            bg = color_background)

        self.rrb2 = tk.Radiobutton(
            self,
            text="2.5",
            padx=5,
            pady=5,
            variable=self.selected_value2,
            command=self.RadiationLevel,
            value=2,
            font=("Verdana", 11),
            bg = color_background)

        self.rrb3 = tk.Radiobutton(
            self,
            text="5",
            padx=5,
            pady=5,
            variable=self.selected_value2,
            command=self.RadiationLevel,
            value=3,
            font=("Verdana", 11),
            bg = color_background)

        self.rrb4 = tk.Radiobutton(
            self,
            text="10",
            padx=5,
            pady=5,
            variable=self.selected_value2,
            command=self.RadiationLevel,
            value=4,
            font=("Verdana", 11),
            bg = color_background)

        self.rrb5 = tk.Radiobutton(
            self,
            text="20",
            padx=5,
            pady=5,
            variable=self.selected_value2,
            command=self.RadiationLevel,
            value=5,
            font=("Verdana", 11),
            bg = color_background)

        self.rrb6 = tk.Radiobutton(
            self,
            text="30",
            padx=5,
            pady=5,
            variable=self.selected_value2,
            command=self.RadiationLevel,
            value=6,
            font=("Verdana", 11),
            bg = color_background)

        self.label2 = tk.Label(
            self,
            text="Choose Cell Type",
            font=("Verdana Bold", 11),
            bg = color_background)

        self.rrrb1 = tk.Radiobutton(
            self,
            text="Wild Type",
            padx=5,
            pady=5,
            variable=self.selected_value3,
            command=self.CellTypee,
            value=1,
            font=("Verdana", 11),
            bg = color_background)

        self.rrrb2 = tk.Radiobutton(
            self,
            text="rad51",
            padx=5,
            pady=5,
            variable=self.selected_value3,
            command=self.CellTypee,
            value=2,
            font=("Verdana", 11),
            bg = color_background)

        for i in [1, 2, 3, 4, 5, 6]:
            self.columnconfigure(i, weight=1, uniform='fred')
        #
        # self.columnconfigure(uniform= 'fred')

        # self.label0.grid(padx = 10, pday = 10)
        self.label2.grid(row=1, column=0)
        self.rrrb1.grid(row=2, column=0)
        self.rrrb2.grid(row=2, column=1)

        self.label0.grid(row=3, column=0)
        self.rb1.grid(row=4, column=0)
        self.rb2.grid(row=4, column=1)
        self.rb3.grid(row=4, column=2)

        self.label1.grid(row=5, column=0)
        self.rrb1.grid(row=6, column=0)
        self.rrb2.grid(row=6, column=1)
        self.rrb3.grid(row=6, column=2)
        self.rrb4.grid(row=6, column=3)
        self.rrb5.grid(row=6, column=4)
        self.rrb6.grid(row=6, column=5)

        # LOGO

        image1 = Image.open(r'C:\Users\dmpalaci\Documents\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\BPS.png')
        self.image1 = image1
        self.image1 = self.image1.resize((215, 100), Image.ANTIALIAS)
        self.test = ImageTk.PhotoImage(self.image1)

        self.label1 = tk.Label(image=self.test)
        self.label1.image = self.test
        self.label1.grid(row = 7, column = 0)


class PLG(tk.Frame):

    
    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)


        self.controller = controller
        self.wentry = tk.Entry(self)
        label = tk.Label(self, text="results")
        self.text = tk.Text(self)

        self.canvas = tk.Canvas(self, width=250, height=250)

        self.canvas.grid()
        self.config(bg="light cyan")
        #######################
        self.lst = tk.Listbox(self, width=10)
        self.lst.grid(row=0)

        self.lst.bind("<<ListboxSelect>>", self.showimg)
        self.lst.bind("<Control-d>", self.delete_item)
        self.insertfiles()
        self.lst.grid(row=0, column=5)
        ######################################

        # restart_button = tk.Button(self, text="Restart", command=self.restart)
        # restart_button.grid()
        # refresh_button = tk.Button(self, text="Refresh", command=self.refresh)
        # refresh_button.grid()


        ### LEGEND
        image1 = Image.open(r'C:\Users\dmpalaci\Documents\Daniel_Master_Directory\AMMPER\AMMPER_NEW\AMMPER\Legend.png')
        self.image1 = image1
        self.image1 = self.image1.resize((255, 185), Image.ANTIALIAS)
        self.test = ImageTk.PhotoImage(self.image1)

        self.label1 = tk.Label(image=self.test)
        self.label1.image = self.test
        self.label1.grid(row=7, column=0)

    def insertfiles(self):
        global pathnn
        print(pathnn)


        for filename in glob.glob(pathnn):
            self.lst.insert(tk.END, filename)

    ####### MIGHT NEED TO CHANGE ORDER
    def delete_item(self, event):
        n = self.lst.curselection()
        os.remove(self.lst.get(n))
        self.lst.delete(n)

    def get_window_size(self):
        if self.winfo_width() > 200 and self.winfo_height() > 30:
            w = self.winfo_width() - 200
            h = self.winfo_height() - 30
        else:
            w = 200
            h = 30
        return w, h

    def showimg(self, event):
        n = self.lst.curselection()
        filename = self.lst.get(n)
        im = Image.open(filename)
        im = im.resize((640,480), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(im)
        w, h = img.width(), img.height()
        self.canvas.image = img

        self.canvas.config(width=w, height=h)
        self.canvas.create_image(0, 0, image=img, anchor="nw")
        self.bind("<Configure>", lambda x: self.showimg(x))

    def restart(self):
        self.refresh()
        self.controller.show_frame("StartPage")

    def refresh(self):
        self.wentry.delete(0, "end")
        self.text.delete("1.0", "end")
        # set focus to any widget except a Text widget so focus doesn't get stuck in a Text widget when page hides
        self.wentry.focus_set()

if __name__ == "__main__":

    pathnn = 'k'
    app = SampleApp()
    print(StartPage.path)
    path = app.frames[StartPage.__name__].RadiationLevel()
    app.mainloop()

    
pathnn = app.frames[StartPage.__name__].RadiationLevel()
print("GLOBAL PATH AFTER")
print(pathnn)

# time.sleep(.5)

app2 = SampleApp2()
app2.mainloop()

# NOTE THAT START AMMPER is only closing the Frame and not the main app ...