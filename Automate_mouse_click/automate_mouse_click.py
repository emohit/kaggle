# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 11:20:22 2017

@author: monarang
"""

import pyautogui
import time

while(1):
    x, y = pyautogui.position()
    positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
    print positionStr
    time.sleep(5)

    

while(1):
    pyautogui.click(1367,1150)
    pyautogui.click(1368,1167)    
    time.sleep(2)