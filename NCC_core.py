#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-to-displacement pipeline (Fourier + Equal-size correlation).
Command-line usage:
    python run_vxy.py ker_size win_size startxy lengthxy basis_fold sta_name pict_file target_file maskurf
All arguments are positional and required.
"""

"""
Usage type 1:
    python NCC_core.py \
        --ker_size INT \
        --win_size INT \
        --startxy INT INT \
        --lengthxy INT INT \
        --basis_fold PATH \
        --sta_name PATH \
        --pict_file PATH \
        --target_file PATH \
        [--maskurf PATH | None]
"""

"""
Usage type 2:
    python NCC_core.py \
        --config PATH
"""

import os
import numpy as np
import cv2
import pandas as pd
import tifffile as tif
from tqdm import tqdm
from scipy.optimize import curve_fit
from numpy.core import asarray, zeros, swapaxes, take
from numpy.core.multiarray import normalize_axis_index
from numpy.fft import _pocketfft_internal as pfi
import argparse
import ast
from pathlib import Path
import sys
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import SubFunction.Picture_size_check as pictcheck

# ------------------------------------------------------------------
# Global configuration (can be turned into CLI flags if needed)
# ------------------------------------------------------------------
judgment_type = ['distance'] # ['distance', 'ci_low']
distance_num = 150
distance_num_second = 500
distance_point_num = 5
distance_point_num_second = 5
ci_low = 0.1
ncctype = 'F-NCC'      # 'S-NCC', 'F-NCC', 'O-NCC-norm', 'O-NCC-sig'
picttype = 'jpg'       # 'jpg', 'tif'

file_name = ["vx_", "vy_", "isprocessed_"]
subfolders = ["dataf", "datae"]
isbreak = True
is_equal_autosave = False
spatial_resolution = 10
# ------------------------------------------------------------------

SYS_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'System_config.txt')
def load_sys_config():
    cfg = {}
    with open(SYS_CFG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, val = map(str.strip, line.split('=', 1))
            try:
                cfg[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                cfg[key] = val
    return cfg
SYS_CFG = load_sys_config()
globals().update({
    'judgment_type': SYS_CFG['judgment_type'],
    'distance_num': SYS_CFG['distance_num'],
    'distance_num_second': SYS_CFG['distance_num_second'],
    'distance_point_num': SYS_CFG['distance_point_num'],
    'distance_point_num_second': SYS_CFG['distance_point_num_second'],
    'ci_low': SYS_CFG['ci_low'],
    'ncctype': SYS_CFG['ncctype'],
    'picttype': SYS_CFG['picttype'],
    'file_name': SYS_CFG['file_name'],
    'subfolders': SYS_CFG['subfolders'],
    'isbreak': SYS_CFG['isbreak'],
    'is_equal_autosave': SYS_CFG['is_equal_autosave'],
    'spatial_resolution': SYS_CFG['spatial_resolution']
})

SYS_CFG = load_sys_config()
# ------------------------------------------------------------------
class matchimg:
    def __init__(self,fold_urf,img_old_urf,img_new_urf,type='F-NCC',picttype='jpg'):
        self.img_old = img_old_urf
        self.img_new = img_new_urf
        self.fold = fold_urf
        self.type = type
        self.picttype=picttype

    def read_img(self):
        if self.picttype=='tif':
            self.img_old = tif.imread(self.fold+self.img_old)
            self.img_new = tif.imread(self.fold+self.img_new)
            # print(self.img_old.shape,self.img_new.shape)
        else:
            self.img_old = cv2.imread(self.fold+self.img_old, cv2.IMREAD_GRAYSCALE)
            self.img_new = cv2.imread(self.fold+self.img_new, cv2.IMREAD_GRAYSCALE)

        if self.type == 'O-NCC-norm':
            tmp_old_x = (cv2.Sobel(self.img_old, cv2.CV_64F, 1, 0, ksize=3)/8).astype(np.uint8)
            tmp_old_y = (cv2.Sobel(self.img_old, cv2.CV_64F, 0, 1, ksize=3)/8).astype(np.uint8)
            tmp_new_x = (cv2.Sobel(self.img_new, cv2.CV_64F, 1, 0, ksize=3)/8).astype(np.uint8)
            tmp_new_y = (cv2.Sobel(self.img_new, cv2.CV_64F, 0, 1, ksize=3)/8).astype(np.uint8)
            self.direction_old  = (tmp_old_x + 1j*tmp_old_y)/np.sqrt(tmp_old_x**2 + tmp_old_y**2+1e-12)
            self.direction_new  = (tmp_new_x + 1j*tmp_new_y)/np.sqrt(tmp_new_x**2 + tmp_new_y**2+1e-12)
            del tmp_old_x, tmp_old_y, tmp_new_x, tmp_new_y

        elif self.type == 'O-NCC-sig':
            tmp_old_x = cv2.Sobel(self.img_old, 8, 1, 0, ksize=3)
            tmp_old_y = cv2.Sobel(self.img_old, 8, 0, 1, ksize=3)
            tmp_new_x = cv2.Sobel(self.img_new, 8, 1, 0, ksize=3)
            tmp_new_y = cv2.Sobel(self.img_new, 8, 0, 1, ksize=3)
            self.direction_old  = np.sign(tmp_old_x + 1j*tmp_old_y)
            self.direction_new  = np.sign(tmp_new_x + 1j*tmp_new_y)
            del tmp_old_x, tmp_old_y, tmp_new_x, tmp_new_y

class img2vxy_first:
    def __init__(self,matchimg,ker_size,win_size,startxy,lengthxy,
                 foldurf,finish_name_urf,err_name_urf,ncctype='F-NCC'):
        self.imgs = matchimg
        self.delta_x = np.zeros((lengthxy[0],lengthxy[1]))
        self.delta_y = np.zeros((lengthxy[0],lengthxy[1]))
        self.lengthxy = lengthxy
        self.ker_size = ker_size
        self.win_size = win_size
        self.startxy = startxy
        self.isprocessed = np.zeros_like(self.delta_x)

        self.foldurf = foldurf
        self.finish_name_urf = finish_name_urf
        self.err_name_urf = err_name_urf

        self.type=ncctype

    # FFT utilities
    #..........................................................
    def raw_fft(self,a, n, axis, is_real, is_forward, inv_norm):   #为了变换结果，这里需要定义浮点数inv_norm
        axis = normalize_axis_index(axis, a.ndim)
        if n is None:
            n = a.shape[axis]

        fct = 1/inv_norm

        if a.shape[axis] != n:
            s = list(a.shape)
            index = [slice(None)]*len(s)
            if s[axis] > n:
                index[axis] = slice(0, n)
                a = a[tuple(index)]
            else:
                index[axis] = slice(0, s[axis])
                s[axis] = n
                z = zeros(s, a.dtype.char)
                z[tuple(index)] = a
                a = z

        if axis == a.ndim-1:
            r = pfi.execute(a, is_real, is_forward, fct)
        else:
            a = swapaxes(a, axis, -1)
            r = pfi.execute(a, is_real, is_forward, fct)
            r = swapaxes(r, axis, -1)
        return r
    def fft(self,a, n=None, axis=-1):   #定义一维傅里叶变换函数
        a = asarray(a)
        if n is None:
            n = a.shape[axis]
        inv_norm = 1
        output = self.raw_fft(a, n, axis, False, True, inv_norm)
        return output
    def ifft(self,a, n=None, axis=-1):
        a = asarray(a)
        if n is None:
            n = a.shape[axis]
        inv_norm = n
        output = self.raw_fft(a, n, axis, False, False, inv_norm)
        return output
    def cook_nd_args(self,a, shape=None, axes=None, invreal=0):
        if shape is None:
            shapeless = 1
            if axes is None:
                shape = list(a.shape)
            else:
                shape = take(a.shape, axes)
        else:
            shapeless = 0
        shape = list(shape)
        if axes is None:
            axes = list(range(-len(shape), 0))
        if invreal and shapeless:
            shape[-1] = (a.shape[axes[-1]] - 1) * 2
        return shape, axes
    def raw_fftnd(self,a, shape=None, axes=None, function=fft):
        a = asarray(a)
        shape, axes = self.cook_nd_args(a, shape, axes)
        itl = list(range(len(axes)))
        itl.reverse()
        for ii in itl:
            a = function(a, n=shape[ii], axis=axes[ii])
        return a
    def imagefft2(self,a, shape=None, axes=(-2, -1)):
        return self.raw_fftnd(a, shape, axes, self.fft)
    def imageifft2(self,a, shape=None, axes=(-2, -1)):
        return self.raw_fftnd(a, shape, axes, self.ifft)
    #..........................................................
    # Correlation kernels
    def fft2CI(self,ker,win):
        win_height, win_width = win.shape
        t_fft = self.imagefft2(ker, shape=(win_height, win_width))
        o_fft = self.imagefft2(win)
        ci = self.imageifft2(np.multiply(o_fft, t_fft.conj()) / (np.abs(np.multiply(o_fft, t_fft.conj())))+1e-12)
        return ci.real


    # Spatial utilities
    def spatialNCC(self,ker,win):
        win_height, win_width = win.shape
        ker_height, ker_width = ker.shape
        ci = np.zeros((win_height-ker_height+1, win_width-ker_width+1))
        for ii in range(win_height-ker_height+1):
            for jj in range(win_width-ker_width+1):
                template = win[ii:ii+ker_height,jj:jj+ker_width]
                meanr, means = np.mean(template), np.mean(ker)
                cor = np.sum((template-meanr)*(ker-means))
                ci[ii,jj] = cor/(np.std(template)*np.std(ker))
        return ci

    # Main functions
    def top_n_indices_2d(self,arr, n):
        flat_indices = np.argpartition(arr.ravel(), -n)[-n:]
        indices = np.unravel_index(flat_indices, arr.shape)
        values = arr[indices]
        return indices[0],indices[1],values

    def is_matchpoint(self,ci,n):
        index_x,index_y,index_ci = self.top_n_indices_2d(ci,n)
        # cimean = np.mean(ci)
        # cistd = np.std(ci)
        # index_ci = (index_ci-cimean)/cistd
        ci_max_idx = np.where(index_ci == np.max(index_ci))[0][0]

        degree = 0
        for ii in range(n):
            if ii==ci_max_idx:
                continue
            if (index_x[ii]-index_x[ci_max_idx])**2 + (index_y[ii]-index_y[ci_max_idx])**2<=5:
                continue
            degree += 1/abs(index_ci[ii]-index_ci[ci_max_idx])
        return degree/n

    def findmax(self,ci):
        def parabola_fit(x, a, b, c, d, e, f):
            return (a*x[0]**2 + b*x[1]**2 + c*x[0]*x[1] + d*x[0] + e*x[1] + f).ravel()

        result = np.where(ci == np.amax(ci))
        result = np.array(list(zip(result[0], result[1]))[0])
        ci_size = ci.shape
        if np.any(np.logical_or(result>=ci_size[0]-2,result<=1)):
            return np.nan,np.nan
        new_ci = ci[result[0]-2:result[0]+3,result[1]-2:result[1]+3]*100
        para_xy = np.indices([5,5])-2
        para_z = new_ci.flatten()
        para_popt, _ = curve_fit(parabola_fit, para_xy, para_z)

        para_A = np.array([[2*para_popt[0],para_popt[2]],[para_popt[2],2*para_popt[1]]])
        para_b = np.array([-para_popt[3],-para_popt[4]])
        para_solvexy = np.linalg.solve(para_A,para_b)

        if para_solvexy[0]**2 + para_solvexy[1]**2>8:
            return np.nan,np.nan
        return result[1]+para_solvexy[1],result[0]+para_solvexy[0]

    def auto_middle_save(self,name_urf):
        df = pd.DataFrame(self.delta_x)
        df.to_csv(self.foldurf+name_urf[0]+'.csv', index=False)
        df = pd.DataFrame(self.delta_y)
        df.to_csv(self.foldurf+name_urf[1]+'.csv', index=False)
        df = pd.DataFrame(self.isprocessed)
        df.to_csv(self.foldurf+name_urf[2]+'.csv', index=False)

    '''
    S-NCC, F-NCC, O-NCC-norm, O-NCC-sig
    '''
    def matchprocess(self,mask=None):

        if self.type == 'S-NCC':
            CIFunc = self.spatialNCC
        else:
            CIFunc = self.fft2CI

        if mask is None:
            mask = np.ones_like(self.imgs.img_old)
        for ii in tqdm(range(self.lengthxy[0])):
            for jj in range(self.lengthxy[1]):
                if not mask[ii,jj]:
                    self.delta_x[ii,jj],self.delta_y[ii,jj] = np.nan,np.nan
                    self.isprocessed[ii,jj] = 1
                    continue
                if self.isprocessed[ii,jj]:
                    continue
                if self.type == 'O-NCC-norm' or self.type == 'O-NCC-sig':
                    imgker = self.imgs.direction_old[self.startxy[0]+ii-self.ker_size//2:self.startxy[0]+ii+self.ker_size//2,
                                                     self.startxy[1]+jj-self.ker_size//2:self.startxy[1]+jj+self.ker_size//2]
                    imgwin = self.imgs.direction_new[self.startxy[0]+ii-self.win_size//2:self.startxy[0]+ii+self.win_size//2,
                                                     self.startxy[1]+jj-self.win_size//2:self.startxy[1]+jj+self.win_size//2]
                else:
                    imgker = self.imgs.img_old[self.startxy[0]+ii-self.ker_size//2:self.startxy[0]+ii+self.ker_size//2,
                                               self.startxy[1]+jj-self.ker_size//2:self.startxy[1]+jj+self.ker_size//2]
                    imgwin = self.imgs.img_new[self.startxy[0]+ii-self.win_size//2:self.startxy[0]+ii+self.win_size//2,
                                               self.startxy[1]+jj-self.win_size//2:self.startxy[1]+jj+self.win_size//2]
                ci = CIFunc(imgker,imgwin)
                if np.isnan(ci[0,0]):
                    self.delta_x[ii,jj],self.delta_y[ii,jj] = np.nan,np.nan
                    # print(f'{self.startxy[0]+ii} , {self.startxy[1]+jj} is missing')
                    self.isprocessed[ii,jj] = 2
                    continue
                if 'distance' in judgment_type:
                    degree = self.is_matchpoint(ci,distance_point_num)
                    if degree > distance_num:
                        self.delta_x[ii,jj],self.delta_y[ii,jj] = np.nan,np.nan
                        self.isprocessed[ii,jj] = 2
                        continue
                if 'ci_low' in judgment_type:
                    if np.nanmax(ci) < ci_low:
                        self.delta_x[ii,jj],self.delta_y[ii,jj] = np.nan,np.nan
                        self.isprocessed[ii,jj] = 2
                        continue
                tmpdx,tmpdy = self.findmax(ci)
                self.delta_x[ii,jj],self.delta_y[ii,jj] = tmpdx,tmpdy
                if np.isnan(tmpdx):
                    self.isprocessed[ii,jj] = 2
                else:
                    self.isprocessed[ii,jj] = 1

    def completeprocess(self,is_after_err=False,past_arr_urf=None,mask=None):
        if is_after_err:
            df = pd.read_csv(self.foldurf+past_arr_urf[0]+'.csv')
            self.delta_x = np.array(df)

            if (self.delta_x.shape[0] != self.lengthxy[0] or
                    self.delta_x.shape[1] != self.lengthxy[1]):
                print('Unfinished task dimensions do not match this object! Please check parameters.')
                print(f'Unfinished task shape: {self.delta_x.shape}')
                print(f'This object shape: {self.lengthxy[0]} , {self.lengthxy[1]}')
                return None

            df = pd.read_csv(self.foldurf+past_arr_urf[1]+'.csv')
            self.delta_y = np.array(df)

            df = pd.read_csv(self.foldurf+past_arr_urf[2]+'.csv')
            self.isprocessed = np.array(df)

        try:
            self.matchprocess(mask)
        except KeyboardInterrupt:
            print('Manually interrupted!')
            self.auto_middle_save(self.err_name_urf)
            print('Temporary save completed.')
            return 0
        except ZeroDivisionError:
            print('Division by zero!')
            self.auto_middle_save(self.err_name_urf)
            print('Temporary save completed.')
            return 0
        except IndexError:
            print('Array out of bounds!')
            self.auto_middle_save(self.err_name_urf)
            print('Temporary save completed.')
            return 0
        else:
            print('Completed!')
            self.delta_x += (-self.win_size // 2 + self.ker_size // 2)
            self.delta_y += (-self.win_size // 2 + self.ker_size // 2)
            self.auto_middle_save(self.finish_name_urf)
            print('Save completed.')
            return 1

    def read_finished_vxy(self,name_urf):
        df = pd.read_csv(self.foldurf+name_urf[0]+'.csv')
        self.delta_x = np.array(df)

        if (self.delta_x.shape[0] != self.lengthxy[0] or
                self.delta_x.shape[1] != self.lengthxy[1]):
            print('Finished task dimensions do not match this object! Please check parameters.')
            print(f'Finished task shape: {self.delta_x.shape}')
            print(f'This object shape: {self.lengthxy[0]} , {self.lengthxy[1]}')
            return None

        df = pd.read_csv(self.foldurf+name_urf[1]+'.csv')
        self.delta_y = np.array(df)

        df = pd.read_csv(self.foldurf+name_urf[2]+'.csv')
        self.isprocessed = np.array(df)

        return True

class img2vxy_end:
    def __init__(self,matchimg,ker_size,win_size,startxy,lengthxy,
                 foldurf,finish_name_urf,err_name_urf,ncctype):
        self.imgs = matchimg

        self.delta_x = np.zeros((lengthxy[0],lengthxy[1]))
        self.delta_y = np.zeros((lengthxy[0],lengthxy[1]))
        self.lengthxy = lengthxy
        self.ker_size = ker_size
        self.win_size = win_size
        self.startxy = startxy
        self.isprocessed = np.zeros_like(self.delta_x)

        self.foldurf = foldurf
        self.finish_name_urf = finish_name_urf
        self.err_name_urf = err_name_urf

        self.type=ncctype


    def read_fourier_vxy(self,out_foldurf,name_urf,is_used_fourier=False,fourier=None):
        if not is_used_fourier:
            df = pd.read_csv(out_foldurf + name_urf[0] + '.csv')
            self.delta_x = np.array(df)
            if self.delta_x.shape[0] != self.lengthxy[0] or self.delta_x.shape[1] != self.lengthxy[1]:
                print('Fourier task dimensions do not match this object! Please check parameters.')
                print(f'Fourier task shape: {self.delta_x.shape}')
                print(f'This object shape: {self.lengthxy[0]} , {self.lengthxy[1]}')
                return None

            df = pd.read_csv(out_foldurf + name_urf[1] + '.csv')
            self.delta_y = np.array(df)

            df = pd.read_csv(out_foldurf + name_urf[2] + '.csv')
            self.isprocessed = np.array(df)
            print('Fourier-processed files imported successfully!')
        else:
            self.delta_x = fourier.delta_x
            if self.delta_x.shape[0] != self.lengthxy[0] or self.delta_x.shape[1] != self.lengthxy[1]:
                print('Fourier task dimensions do not match this object! Please check parameters.')
                print(f'Fourier task shape: {self.delta_x.shape}')
                print(f'This object shape: {self.lengthxy[0]} , {self.lengthxy[1]}')
                return None
            self.delta_y = fourier.delta_y
            self.isprocessed = fourier.isprocessed
            print('Fourier-processed memory referenced successfully!')

    def save_range_data(self,dxrange,dyrange):
        nanidx = np.where(np.logical_or(np.logical_or(self.delta_x<dxrange[0],self.delta_x>dxrange[1]),
                                       np.logical_or(self.delta_y<dyrange[0],self.delta_y>dyrange[1])))
        self.delta_x[nanidx] = np.nan
        self.delta_y[nanidx] = np.nan
        self.isprocessed[nanidx] = 2

    def choose_nearest_distance(self,ii,jj):
        radius = 1
        tmpstate = 0
        maxiter = 0

        while not tmpstate:
            maxiter+=1
            tmp_isprocessed = self.isprocessed[max(0,ii-radius):min(self.isprocessed.shape[0],ii+radius+1),
                                               max(0,jj-radius):min(self.isprocessed.shape[1],jj+radius+1)]
            tmp_dx = self.delta_x[max(0,ii-radius):min(self.isprocessed.shape[0],ii+radius+1),
                                  max(0,jj-radius):min(self.isprocessed.shape[1],jj+radius+1)]
            if 2*tmp_isprocessed.shape[0]*tmp_isprocessed.shape[1] - np.sum(tmp_isprocessed) > 5 and (not np.isnan(tmp_dx).all()):
                tmpstate = 1
            else:
                radius += 1
            if maxiter>1000:
                return 0,0

        tmp_isprocessed = self.isprocessed[max(0,ii-radius):min(self.isprocessed.shape[0],ii+radius+1),
                                           max(0,jj-radius):min(self.isprocessed.shape[1],jj+radius+1)]

        tmp_idx = np.where(tmp_isprocessed==1)

        tmp_dx = self.delta_x[max(0,ii-radius):min(self.isprocessed.shape[0],ii+radius+1),
                              max(0,jj-radius):min(self.isprocessed.shape[1],jj+radius+1)]
        tmp_dx = tmp_dx[tmp_idx[0],tmp_idx[1]]
        tmp_ave_dx = np.nanmedian(tmp_dx)

        tmp_dy = self.delta_y[max(0,ii-radius):min(self.isprocessed.shape[0],ii+radius+1),
                              max(0,jj-radius):min(self.isprocessed.shape[1],jj+radius+1)]
        tmp_dy = tmp_dy[tmp_idx[0],tmp_idx[1]]
        tmp_ave_dy = np.nanmedian(tmp_dy)

        return int(tmp_ave_dx),   int(tmp_ave_dy)

    def equalCI(self,img, kernel):
        ker_fft = np.fft.fft2(kernel)
        win_fft = np.fft.fft2(img)
        tmp = np.multiply(ker_fft,win_fft.conj())
        tmp = tmp/(np.abs(tmp)+1e-12)
        tmpCImat = np.fft.ifft2(tmp)
        tmpCImat = tmpCImat.real
        CIshape = tmpCImat.shape[0]

        newCImat = np.zeros_like(tmpCImat)
        newCImat[CIshape//2:CIshape,CIshape//2:CIshape] = tmpCImat[:CIshape//2,:CIshape//2]
        newCImat[:CIshape//2,CIshape//2:CIshape] = tmpCImat[CIshape//2:CIshape,:CIshape//2]
        newCImat[CIshape//2:CIshape,:CIshape//2] = tmpCImat[:CIshape//2,CIshape//2:CIshape]
        newCImat[:CIshape//2,:CIshape//2] = tmpCImat[CIshape//2:CIshape,CIshape//2:CIshape]
        return newCImat

    def top_n_indices_2d(self,arr, n):
        flat_indices = np.argpartition(arr.ravel(), -n)[-n:]
        indices = np.unravel_index(flat_indices, arr.shape)
        values = arr[indices]
        return indices[0],indices[1],values

    def is_matchpoint(self,ci,n):
        index_x,index_y,index_ci = self.top_n_indices_2d(ci,n)
        # cimean = np.mean(ci)
        # cistd = np.std(ci)
        # index_ci = (index_ci-cimean)/cistd
        ci_max_idx = np.where(index_ci == np.max(index_ci))[0][0]

        degree = 0
        for ii in range(n):
            if ii==ci_max_idx:
                continue
            if (index_x[ii]-index_x[ci_max_idx])**2 + (index_y[ii]-index_y[ci_max_idx])**2<=5:
                continue
            degree += 1/abs(index_ci[ii]-index_ci[ci_max_idx])
        return degree/n

    def findmax(self,ci):
        def parabola_fit(x, a, b, c, d, e, f):
            return (a*x[0]**2 + b*x[1]**2 + c*x[0]*x[1] + d*x[0] + e*x[1] + f).ravel()

        result = np.where(ci == np.amax(ci))
        result = np.array(list(zip(result[0], result[1]))[0])
        ci_size = ci.shape
        if np.any(np.logical_or(result>=ci_size[0]-2,result<=1)):
            return np.nan,np.nan
        new_ci = ci[result[0]-2:result[0]+3,result[1]-2:result[1]+3]*100
        para_xy = np.indices([5,5])-2
        para_z = new_ci.flatten()
        para_popt, _ = curve_fit(parabola_fit, para_xy, para_z)

        para_A = np.array([[2*para_popt[0],para_popt[2]],[para_popt[2],2*para_popt[1]]])
        para_b = np.array([-para_popt[3],-para_popt[4]])
        para_solvexy = np.linalg.solve(para_A,para_b)

        if para_solvexy[0]**2 + para_solvexy[1]**2>8:
            return np.nan,np.nan
        return result[1]+para_solvexy[1]-self.ker_size//2,result[0]+para_solvexy[0]-self.ker_size//2


    def single_match(self,ii,jj):
        if self.isprocessed[ii,jj]==1:
            return None
        dy,dx = self.choose_nearest_distance(ii,jj)
        if self.type == 'O-NCC-norm' or self.type == 'O-NCC-sig':
            imgker = self.imgs.direction_old[self.startxy[0]+ii-self.ker_size//2:self.startxy[0]+ii+self.ker_size//2,
                                             self.startxy[1]+jj-self.ker_size//2:self.startxy[1]+jj+self.ker_size//2]
            imgwin = self.imgs.direction_new[self.startxy[0]+ii-self.ker_size//2+dx:self.startxy[0]+ii+self.ker_size//2+dx,
                                             self.startxy[1]+jj-self.ker_size//2+dy:self.startxy[1]+jj+self.ker_size//2+dy]
        else:
            imgker = self.imgs.img_old[self.startxy[0]+ii-self.ker_size//2:self.startxy[0]+ii+self.ker_size//2,
                                       self.startxy[1]+jj-self.ker_size//2:self.startxy[1]+jj+self.ker_size//2]
            imgwin = self.imgs.img_new[self.startxy[0]+ii-self.ker_size//2+dx:self.startxy[0]+ii+self.ker_size//2+dx,
                                       self.startxy[1]+jj-self.ker_size//2+dy:self.startxy[1]+jj+self.ker_size//2+dy]
        ci = self.equalCI(imgker,imgwin)
        if np.isnan(ci[0,0]):
            return None
        if 'distance' in judgment_type:
            degree = self.is_matchpoint(ci,distance_point_num_second)
            if degree > distance_num_second:
                return None
        if 'ci_low' in judgment_type:
            if np.nanmax(ci) < ci_low:
                return None
        self.delta_x[ii,jj],self.delta_y[ii,jj] = self.findmax(ci)
        if np.isnan(self.delta_x[ii,jj]):
            return None
        else:
            self.delta_x[ii,jj]+=dy
            self.delta_y[ii,jj]+=dx
            self.isprocessed[ii,jj] = 1
            return True

    def auto_middle_save(self,name_urf):
        df = pd.DataFrame(self.delta_x)
        df.to_csv(self.foldurf+name_urf[0]+'.csv', index=False)
        df = pd.DataFrame(self.delta_y)
        df.to_csv(self.foldurf+name_urf[1]+'.csv', index=False)
        df = pd.DataFrame(self.isprocessed)
        df.to_csv(self.foldurf+name_urf[2]+'.csv', index=False)

    def ranking_sequence(self,isautosave = True,mask=None):
        if mask is None:
            mask = np.ones_like(self.delta_x)
        for ii in tqdm(np.flip(np.arange(self.delta_x.shape[0] // 2))):
            for jj in np.flip(np.arange(self.delta_x.shape[1] // 2)):
                if mask[ii, jj] == 0:
                    continue
                self.single_match(ii, jj)
        if isautosave:
            self.auto_middle_save(self.err_name_urf)
            print('Auto-backup completed each time!')
        for ii in tqdm(np.arange(self.delta_x.shape[0] // 2, self.delta_x.shape[0])):
            for jj in np.flip(np.arange(self.delta_x.shape[1] // 2)):
                if mask[ii, jj] == 0:
                    continue
                self.single_match(ii, jj)
        if isautosave:
            self.auto_middle_save(self.err_name_urf)
            print('Auto-backup completed each time!')
        for ii in tqdm(np.flip(np.arange(self.delta_x.shape[0] // 2))):
            for jj in np.arange(self.delta_x.shape[1] // 2, self.delta_x.shape[1]):
                if mask[ii, jj] == 0:
                    continue
                self.single_match(ii, jj)
        if isautosave:
            self.auto_middle_save(self.err_name_urf)
            print('Auto-backup completed each time!')
        for ii in tqdm(np.arange(self.delta_x.shape[0] // 2, self.delta_x.shape[0])):
            for jj in np.arange(self.delta_x.shape[1] // 2, self.delta_x.shape[1]):
                if mask[ii, jj] == 0:
                    continue
                self.single_match(ii, jj)

    def completeprocess(self, isautosave=True, is_after_err=False, past_arr_urf=None, mask=None):
        if self.type == 'S-NCC':
            print('Spatial NCC does not need extra processing')
            return 1
        if is_after_err:
            df = pd.read_csv(self.foldurf + past_arr_urf[0] + '.csv')
            self.delta_x = np.array(df)

            if self.delta_x.shape[0] != self.lengthxy[0] or self.delta_x.shape[1] != self.lengthxy[1]:
                print('Unfinished task dimensions do not match this object! Please check parameters.')
                print(f'Unfinished task shape: {self.delta_x.shape}')
                print(f'This object shape: {self.lengthxy[0]} , {self.lengthxy[1]}')
                return None

            df = pd.read_csv(self.foldurf + past_arr_urf[1] + '.csv')
            self.delta_y = np.array(df)

            df = pd.read_csv(self.foldurf + past_arr_urf[2] + '.csv')
            self.isprocessed = np.array(df)

        try:
            self.ranking_sequence(isautosave, mask=mask)
        except KeyboardInterrupt:
            print('Manually interrupted!')
            self.auto_middle_save(self.err_name_urf)
            print('Temporary save completed')
            return 0
        except ZeroDivisionError:
            print('Division by zero!')
            self.auto_middle_save(self.err_name_urf)
            print('Temporary save completed')
            return 0
        except IndexError:
            print('Array out of bounds!')
            self.auto_middle_save(self.err_name_urf)
            print('Temporary save completed')
            return 0
        else:
            print('Completed!')
            self.auto_middle_save(self.finish_name_urf)
            print('Save completed')
            return 1

    def read_finished_vxy(self, name_urf):
        df = pd.read_csv(self.foldurf + name_urf[0] + '.csv')
        self.delta_x = np.array(df)

        if self.delta_x.shape[0] != self.lengthxy[0] or self.delta_x.shape[1] != self.lengthxy[1]:
            print('Finished task dimensions do not match this object! Please check parameters.')
            print(f'Finished task shape: {self.delta_x.shape}')
            print(f'This object shape: {self.lengthxy[0]} , {self.lengthxy[1]}')
            return None

        df = pd.read_csv(self.foldurf + name_urf[1] + '.csv')
        self.delta_y = np.array(df)

        df = pd.read_csv(self.foldurf + name_urf[2] + '.csv')
        self.isprocessed = np.array(df)

class img2vxy:
    def __init__(self,ker_size,win_size,startxy,lengthxy,
                 img_fold_urf,img_old_urf,img_new_urf,
                 foldurf_fourier,finish_name_urf_fourier,err_name_urf_fourier,
                 foldurf_equal,finish_name_urf_equal,err_name_urf_equal,ncctype='F-NCC',picttype='jpg'
                 ):
        self.foldurf_fourier = foldurf_fourier
        self.finish_name_urf_fourier = finish_name_urf_fourier
        self.err_name_urf_fourier = err_name_urf_fourier

        self.foldurf_equal = foldurf_equal
        self.finish_name_urf_equal = finish_name_urf_equal
        self.err_name_urf_equal = err_name_urf_equal

        self.imgs = matchimg(img_fold_urf,img_old_urf,img_new_urf,ncctype,picttype)
        self.imgs.read_img()

        self.first = img2vxy_first(self.imgs,ker_size,win_size,startxy,lengthxy,
                                       foldurf_fourier,finish_name_urf_fourier,err_name_urf_fourier,ncctype)

        self.end = img2vxy_end(self.imgs,ker_size,win_size,startxy,lengthxy,
                                       foldurf_equal,finish_name_urf_equal,err_name_urf_equal,ncctype)

    def process_fourier(self, mask=None):
        if os.path.exists(self.foldurf_fourier + self.finish_name_urf_fourier[0] + '.csv'):
            print('Reading Fourier finished files!')
            if self.first.read_finished_vxy(self.finish_name_urf_fourier) is not None:
                print('Fourier processed files loaded successfully!')
                return 1
            else:
                return 0
        elif os.path.exists(self.foldurf_fourier + self.err_name_urf_fourier[0] + '.csv'):
            print('Reading Fourier interrupted files!')
            state = self.first.completeprocess(True, self.err_name_urf_fourier, mask=mask)
            return state
        else:
            print('Starting fresh Fourier processing!')
            state = self.first.completeprocess(mask=mask)
            return state

    def process_equal(self, day, isautosave=True, is_used_fourier=False, mask=None):
        if os.path.exists(self.foldurf_equal + self.finish_name_urf_equal[0] + '.csv'):
            print('Reading Equal finished files!')
            if self.end.read_finished_vxy(self.finish_name_urf_equal) is not None:
                print('Equal processed files loaded successfully!')
                return 1
            else:
                return 0
        elif os.path.exists(self.foldurf_equal + self.err_name_urf_equal[0] + '.csv'):
            print('Reading Equal interrupted files!')
            state = self.end.completeprocess(
                isautosave=isautosave,
                is_after_err=True,
                past_arr_urf=self.err_name_urf_equal,
                mask=mask
            )
            return state
        else:
            print('Starting fresh Equal processing!')
            self.end.read_fourier_vxy(
                self.foldurf_fourier,
                self.finish_name_urf_fourier,
                is_used_fourier,
                self.first
            )
            self.end.save_range_data([-1 * day, 3 * day], [-1 * day, 3 * day])
            state = self.end.completeprocess(isautosave=isautosave, mask=mask)
            return state

class process_all:
    def __init__(self,basis_fold_urf,sta_name,file_name,pict_file,target_file,
                 ker_size,win_size,startxy,lengthxy,stride=1):
        self.basis_fold_urf = basis_fold_urf
        self.sta_name = sta_name
        self.file_name = file_name
        self.pict_file = pict_file
        self.target_file = target_file

        self.lengthxy = lengthxy
        self.ker_size = ker_size
        self.win_size = win_size
        self.startxy = startxy

        self.stride = stride

        self.initialization()

    def initialization(self):
        if not os.path.exists(self.basis_fold_urf):
            os.makedirs(self.basis_fold_urf)
        if not os.path.exists(self.target_file):
            os.makedirs(self.target_file)
        for sub in subfolders:
            sub_path = os.path.join(self.basis_fold_urf, sub)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
        file_name, file_ext = os.path.splitext(self.sta_name)
        csv_path = f"{file_name}.csv"
        if os.path.exists(csv_path):
            self.sta_name = csv_path
            df = pd.read_csv(self.sta_name)
            self.all_state_list = np.array(df)
            return None
        try:
            if file_ext.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(self.sta_name)
                self.all_state_list = np.array(df)
                df.to_csv(csv_path, index=False)
                self.sta_name = csv_path
            elif file_ext.lower() == '.txt':
                df = pd.read_csv(self.sta_name, sep='\t')
                self.all_state_list = np.array(df)
                df.to_csv(csv_path, index=False)
                self.sta_name = csv_path
            else:
                raise ValueError("Unsupported File Format!")
        except Exception:
            print(f"Please confirm if the **statistics** file format is correct.")

    def finalize(self,v_str,day):
        shutil.copy2(
            self.basis_fold_urf + subfolders[1] + '/' + self.file_name[0] + v_str + '_e' + '.csv',
            self.target_file + '/' + self.file_name[0] + v_str + '.csv'
        )
        shutil.copy2(
            self.basis_fold_urf + subfolders[1] + '/' + self.file_name[1] + v_str + '_e' + '.csv',
            self.target_file + '/' + self.file_name[1] + v_str + '.csv'
        )

        tmpvx = np.array(pd.read_csv(self.target_file + '/' + self.file_name[0] + v_str + '.csv')) / day * spatial_resolution
        tmpvy = -np.array(pd.read_csv(self.target_file + '/' + self.file_name[1] + v_str + '.csv')) / day * spatial_resolution

        if self.stride > 1:

            tmp = tmpvx[::self.stride, ::self.stride]
            df = pd.DataFrame(tmp)
            df.to_csv(self.target_file + '/' + self.file_name[0] + v_str + '.csv', index=False)

            tmp = tmpvy[::self.stride, ::self.stride]
            df = pd.DataFrame(tmp)
            df.to_csv(self.target_file + '/' + self.file_name[1] + v_str + '.csv', index=False)



    def forward(self, type, isbreak=True, is_equal_autosave=True, mask=None, ncctype='F-NCC', picttype='jpg'):
        if type == 'dfe':
            for ii in range(self.all_state_list.shape[0]):
                if self.all_state_list[ii, 4] != 0:
                    continue
                elif self.all_state_list[ii, 3] != 0:
                    v_last = str(self.all_state_list[ii, 0])
                    v_end = str(self.all_state_list[ii, 1])
                    v_str = v_last + '-' + v_end
                    print(f'Starting matching for {v_str}!')
                    finish_name_urf_fourier = [
                        self.file_name[0] + v_str + '_f',
                        self.file_name[1] + v_str + '_f',
                        self.file_name[2] + v_str + '_f'
                    ]
                    err_name_urf_fourier = [
                        'err_' + self.file_name[0] + v_str + '_f',
                        'err_' + self.file_name[1] + v_str + '_f',
                        'err_' + self.file_name[2] + v_str + '_f'
                    ]
                    finish_name_urf_equal = [
                        self.file_name[0] + v_str + '_e',
                        self.file_name[1] + v_str + '_e',
                        self.file_name[2] + v_str + '_e'
                    ]
                    err_name_urf_equal = [
                        'err_' + self.file_name[0] + v_str + '_e',
                        'err_' + self.file_name[1] + v_str + '_e',
                        'err_' + self.file_name[2] + v_str + '_e'
                    ]
                    test = img2vxy(
                        self.ker_size, self.win_size, self.startxy, self.lengthxy,
                        self.pict_file, v_last + '.' + picttype, v_end + '.' + picttype,
                                        self.basis_fold_urf + subfolders[0] + '/', finish_name_urf_fourier, err_name_urf_fourier,
                                        self.basis_fold_urf + subfolders[1] + '/', finish_name_urf_equal, err_name_urf_equal,
                        ncctype, picttype
                    )
                    state = test.process_equal(
                        self.all_state_list[ii, 2],
                        isautosave=is_equal_autosave,
                        mask=mask
                    )

                    if state:
                        df = pd.read_csv(self.sta_name)
                        df.iloc[ii, 4] = 1
                        df.to_csv(self.sta_name, index=False)

                        print(f'Matching for {v_str} completed!')
                    else:
                        print(f'Matching for {v_str} interrupted!')
                        if isbreak:
                            break
                        else:
                            continue
                else:
                    v_last = str(self.all_state_list[ii, 0])
                    v_end = str(self.all_state_list[ii, 1])
                    v_str = v_last + '-' + v_end
                    print(f'Starting matching for {v_str}!')
                    finish_name_urf_fourier = [
                        self.file_name[0] + v_str + '_f',
                        self.file_name[1] + v_str + '_f',
                        self.file_name[2] + v_str + '_f'
                    ]
                    err_name_urf_fourier = [
                        'err_' + self.file_name[0] + v_str + '_f',
                        'err_' + self.file_name[1] + v_str + '_f',
                        'err_' + self.file_name[2] + v_str + '_f'
                    ]
                    finish_name_urf_equal = [
                        self.file_name[0] + v_str + '_e',
                        self.file_name[1] + v_str + '_e',
                        self.file_name[2] + v_str + '_e'
                    ]
                    err_name_urf_equal = [
                        'err_' + self.file_name[0] + v_str + '_e',
                        'err_' + self.file_name[1] + v_str + '_e',
                        'err_' + self.file_name[2] + v_str + '_e'
                    ]
                    test = img2vxy(
                        self.ker_size, self.win_size, self.startxy, self.lengthxy,
                        self.pict_file, v_last + '.' + picttype, v_end + '.' + picttype,
                                        self.basis_fold_urf + subfolders[0] + '/', finish_name_urf_fourier,
                        err_name_urf_fourier,
                                        self.basis_fold_urf + subfolders[1] + '/', finish_name_urf_equal,
                        err_name_urf_equal,
                        ncctype, picttype
                    )
                    state = test.process_fourier(mask)
                    if state:
                        df = pd.read_csv(self.sta_name)
                        df.iloc[ii, 3] = 1
                        df.to_csv(self.sta_name, index=False)
                        print(f'Matching for {v_str} completed!')
                    else:
                        print(f'Matching for {v_str} interrupted!')
                        if isbreak:
                            break
                        else:
                            continue

                    state = test.process_equal(
                        self.all_state_list[ii, 2],
                        isautosave=is_equal_autosave,
                        is_used_fourier=True,
                        mask=mask
                    )
                    if state:
                        df = pd.read_csv(self.sta_name)
                        df.iloc[ii, 4] = 1
                        df.to_csv(self.sta_name, index=False)
                        self.finalize(v_str,self.all_state_list[ii, 2])
                        print(f'Matching for {v_str} completed!')
                    else:
                        print(f'Matching for {v_str} interrupted!')
                        if isbreak:
                            break
                        else:
                            continue
# ------------------------------------------------------------------

# ------------------------------------------------------------------
def load_mask(mask_path):
    if mask_path is None or mask_path.lower() == 'none':
        return None

    ext = os.path.splitext(mask_path)[1].lower()
    if ext == '.csv':
        return np.array(pd.read_csv(mask_path, header=None))
    elif ext in ('.tif', '.tiff'):
        return tif.imread(mask_path)
    elif ext in ('.jpg', '.jpeg', '.png'):
        return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError('Unsupported mask file format! Only .csv, .tif/.tiff, .jpg/.jpeg/.png are supported.')

def build_stride_mask(lengthxy, stride):
    h, w = lengthxy
    ii, jj = np.ogrid[:h, :w]
    mask = ((ii % stride == 0) & (jj % stride == 0)).astype(np.uint8)
    return mask
def combine_masks(base_mask, stride_mask):
    if base_mask is None:
        return stride_mask
    if base_mask.shape != stride_mask.shape:
        raise ValueError('Mask shape mismatch: {} vs {}'.format(base_mask.shape, stride_mask.shape))
    return base_mask & stride_mask

def parse_txt_config(path):
    cfg = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            key, val = key.strip(), val.strip()
            # bool / int / float / list
            if val.lower() in ('true', 'false'):
                cfg[key] = val.lower() == 'true'
            else:
                parts = val.split()
                try:
                    if len(parts) > 1:
                        cfg[key] = [int(p) if p.isdigit() else float(p) for p in parts]
                    else:
                        cfg[key] = int(val) if val.isdigit() else float(val)
                except ValueError:
                    cfg[key] = val
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Batch matching via CLI or txt config.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config', type=str,
                       help='Path to txt config file (key = value per line)')
    # The remaining parameters remain compatible and only take effect without --config
    parser.add_argument('--ker_size', type=int)
    parser.add_argument('--win_size', type=int)
    parser.add_argument('--stride', type=int, default=1,
                        help='stride for matching (default=1)')
    parser.add_argument('--startxy', type=int, nargs=2)
    parser.add_argument('--lengthxy', type=int, nargs=2)
    parser.add_argument('--basis_fold', type=str)
    parser.add_argument('--sta_name', type=str)
    parser.add_argument('--pict_file', type=str)
    parser.add_argument('--target_file', type=str)
    parser.add_argument('--maskurf', type=str, default='None')
    return parser.parse_args()
# ------------------------------------------------------------------

def main():
    args = parse_args()
    if args.config:
        cfg = parse_txt_config(args.config)
    else:
        cfg = vars(args)

    def get(k, default=None):
        return cfg.get(k, default)

    ker_size = int(get('ker_size'))
    win_size = int(get('win_size'))
    stride = int(get('stride',1))
    startxy = list(map(int, get('startxy'))); startxy = [startxy[1], startxy[0]]
    lengthxy = list(map(int, get('lengthxy'))); lengthxy = [lengthxy[1], lengthxy[0]]
    basis_fold = str(get('basis_fold'))
    sta_name = str(get('sta_name'))
    pict_file = str(get('pict_file'))
    target_file = str(get('target_file'))
    mask_path = str(get('maskurf', 'None'))

    mask = load_mask(mask_path)

    # Check the format and the size of the pictures
    pictcheck.check_uniform(pict_file)

    if mask is not None and mask.shape != (lengthxy[0], lengthxy[1]):
        raise ValueError('The mask shape is not consistent with the prescribed size of the result!')
    if stride != 1:
        stride_mask = build_stride_mask(lengthxy, stride)
        mask = combine_masks(mask, stride_mask)
    else:
        stride_mask = None
        mask = mask

    start = process_all(
        basis_fold_urf=basis_fold + '/',
        sta_name=sta_name,
        file_name=file_name,
        pict_file=pict_file + '/',
        target_file=target_file + '/',
        ker_size=ker_size,
        win_size=win_size,
        startxy=startxy,
        lengthxy=lengthxy,
        stride=stride
    )

    start.forward(
        'dfe',
        isbreak=isbreak,
        is_equal_autosave=is_equal_autosave,
        ncctype=ncctype,
        picttype=picttype,
        mask=mask
    )

# ------------------------------------------------------------------
if __name__ == '__main__':

    main()
