# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:58:53 2016

@author: monarang
"""

from bs4 import BeautifulSoup
from PIL import Image
import requests
from StringIO import StringIO

s = requests.Session()
first_page = s.get("http://www.bsesdelhi.com/bsesdelhi/caVerification4Pay.do")
