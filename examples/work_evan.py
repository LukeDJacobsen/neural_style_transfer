#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 10:32:44 2021

@author: lukejacobsen
"""

import os

os.chdir("/Users/lukejacobsen/Documents/github/neural_style_transfer")
exec(open('main_functions.py').read())

neural_style_transfer(base_image_path='images/work.jpg', 
                      style_reference_image_path = "images/evan.jpg", 
                      total_variation_weight = 1e-6,
                      style_weight = 1e-4,
                      content_weight = 2.5e-8,
                      img_nrows = 400,
                      opt_decay_rate = 0.96,
                      iterations = 10, 
                      how_often_print_save = 5,
                      outdir = 'images/generated/evan_work')