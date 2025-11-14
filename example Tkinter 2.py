import tkinter as tk
from tkinter.filedialog import askdirectory

def open_file():
    filepath = askdirectory()
    return(print(filepath))

def save_file():
    filepath = askdirectory()
    return(print(filepath))

window = tk.Tk()
window.title("Simple Text Editor")
window.rowconfigure(0, minsize=800, weight=1)
window.columnconfigure(1, minsize=800, weight=1)

var1 = tk.IntVar()

fr_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
btn_open = tk.Button(fr_buttons, text="Open", command=open_file)
btn_save = tk.Button(fr_buttons, text="Save As...", command=save_file)
button = tk.Checkbutton(window, text="male", variable=var1).grid(row=0, sticky=W)

btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
btn_save.grid(row=1, column=0, sticky="ew", padx=5)

fr_buttons.grid(row=0, column=0, sticky="ns")
window.mainloop()