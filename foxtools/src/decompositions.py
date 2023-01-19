# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 20:02:11 2021

@author: foxel
"""

import matplotlib.pyplot as plt
import numpy as np
import math

if __name__ == "__main__":
    import fox_io
else:
    from foxtools.src import fox_io

def plot_eigenvectors(pcComp, xVals, pcNum, fpath, xlims = [380,780]):
    for i, eigenvector in zip(np.arange(pcNum), pcComp):
        if i < max(math.floor(pcNum/2), 5) :  
            plt.plot(xVals, eigenvector, label='eigv'+str(i+1))
        else:
            plt.plot(xVals, eigenvector, '--', label='eigv'+str(i+1))
            
    plt.legend(loc='upper right')
    plt.title('Eigenvectors')
    plt.xlabel('wavelength')
    plt.ylabel('coefficient')
    plt.xlim(xlims)
    #plt.ylim([-0.1, 0.1]) #
    plt.savefig(fpath + 'eigenvectors' + str(pcNum) +'.jpg')
    plt.show() 

def show_reduced_subimages(hsi_image_list, decom, fpath='', num_subimages = 4):
    for i in range(len(hsi_image_list)):
        img = decom.transform(fox_io.flatten_image(hsi_image_list[i]).transpose())
        img = img.reshape((hsi_image_list[i].shape[0], hsi_image_list[i].shape[1]))
        for j in range(num_subimages):
            subimage = img[:,:,j]
            img = (subimage - np.min(subimage)) / (np.max(subimage) - np.min(subimage))
            fox_io.show_image(subimage, 'pc' + str(j) + '(Im' + str(i) + ')', True, fpath)
                
from sklearn.decomposition import PCA
import os.path

def decompose(hsiList, method = 'pca', n_components=10):    
    if hsiList[0].ndim() == 3: 
        stacked = fox_io.flatten_images(hsiList)
    else: 
        stacked = hsiList
    
    print("Total pixels for  fitting: ", stacked.shape[0], 'pixels')

    if method == 'pca': 
        decom = PCA(n_components=n_components)
        decom.fit(stacked)
    
    print("Finished dimension reduction.")
    return decom 

def prep_decomp_figures(decom, hsiList, method = 'pca', savefolder='', rangeLimits = [420, 730]):
    cur_savedir = os.path.join(fox_io.get_savedir(), 'pca', savefolder)
    fox_io.makedir(cur_savedir)
    
    if method == 'pca':
        explained_vals = decom.explained_variance_ratio_
        singular_vals = decom.singular_values_
        print("Explained variance:", explained_vals)
        print("Singular values:", singular_vals)
        
        plt.figure(0)
        fox_io.simple_plot(explained_vals, "Explained variance", "pc number", "explained percentage", cur_savedir)
        plt.figure(1)
        fox_io.simple_plot(singular_vals, "Singular Values", "pc number", "value", cur_savedir)
        
        w = np.arange(rangeLimits[0], rangeLimits[1]+1)
        plt.figure(2)
        plot_eigenvectors(decom.components_, w, 10, cur_savedir)
        plt.figure(3)
        plot_eigenvectors(decom.components_, w, 3, cur_savedir)
        
        plt.figure(4)
        show_reduced_subimages(hsiList, decom, cur_savedir, num_subimages=4)
