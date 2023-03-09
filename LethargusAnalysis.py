"""
Lethargus analysis
20230308
Author: Shinichi Miyazaki

This script enable lethargus analysis for imagesubtraction data
"""

import os
import tkinter.filedialog
import pandas as pd
from functions.myfunctions import lethargus_analyzer

def main():
    # select file
    root = tkinter.Tk()
    root.withdraw()
    filepath = tkinter.filedialog.askopenfilename()
    os.chdir(os.path.dirname(filepath))

    # Load data and make bodysize_list
    data = pd.read_csv(filepath)
    row_number = data.shape[1]
    body_size = int(input('bodysize (pixel) を入力 : '))

    # Lethargus analysis
    lethargus_analyzer(data, body_size)

if __name__ == '__main__':
    main()