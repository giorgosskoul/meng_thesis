# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:12:18 2023

@author: Στεφάνια
"""
from math import sqrt
import numpy as np
from PIL import Image


# QUADRATIC TRANSFORMATION: 
    
def quadratic_transform(Image):
    # - Transforms a non quadratic 2D image to a quadratic.
    ## Image must be in dBs
    ### Dimensions: Rows x Columns = Total Pixels,
    ### Quadratic Dimensions: N1 x N1 = Total Pixels.
    ### Equation: N1=sqrt(Total Pixels), N1 is integer.
    #--------------------------------------------------#
    Rows=float(Image.shape[0])
    Columns=float(Image.shape[1]) 
    C=Columns # new columns with zero-padding
    N1=sqrt(Rows*C) 
    while((N1%1.0!=0.0) or C%1.0!=0.0):
        C=C+1
        N1=sqrt(Rows*C) 
    n_zeros=int(C-Columns)
    return np.reshape(np.pad(Image,((0,0),(0,n_zeros)),"minimum"),(int(N1),int(N1)))

# Resize with PIL

def resize_transform(arr,width,height,resampling=Image.BILINEAR):
    # receives a numpy array input
    ## Image must be in dBs
    ### resampling can be Image.NEAREST,
    ### Image.BOX, Image.BILINEAR, Image.HAMMING,
    ### Image.BICUBIC, Image.LANCZOS.
    ##### As default resampling we use Image.BILINEAR
    ### ---------------------------------------------
    # Convert Numpy array to a PIL Image 
    img = Image.fromarray(arr)
    # Resize the image to a new size
    resized_image = img.resize((width,height),resample=resampling)
    # Convert the PIL image back to a NumPy array
    resized_arr = np.asarray(resized_image)
    return resized_arr
    
