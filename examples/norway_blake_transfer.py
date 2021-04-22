#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 20:42:19 2021

@author: lukejacobsen
"""
import os

os.chdir("/Users/lukejacobsen/Documents/neural_style_transfer")
exec(open('main_functions.py').read())

neural_style_transfer(base_image_path='images/me.jpg', 
                      style_reference_image_path = "images/quentin_blake.jpg", 
                      total_variation_weight = 1e-6,
                      style_weight = 1e-4,
                      content_weight = 2.5e-8,
                      img_nrows = 400,
                      opt_decay_rate = 0.96,
                      iterations = 1000, 
                      how_often_print_save = 100,
                      outdir = 'images/generated/norway_blake')