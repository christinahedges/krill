'''
Creates images of Kepler module large scale structure
'''

import k2movie
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.stats import LombScargle
import astropy.units as u
from tqdm import tqdm
import pandas as pd

from contextlib import contextmanager
import warnings
import sys
@contextmanager
def silence():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout




class krill(object):
    '''Object to hold each channels data'''
    def __init__(self,
                campaign = 1,
                cadence = None,
                dir = '/Users/ch/K2/projects/krill/',
                database_dir = '/Volumes/cupertino/database/',
                verbose = False):

        self.campaign = campaign
        self.cadence = cadence
        self.dir = dir
        self.database_dir = database_dir
        self.verbose = verbose

    def build_cad(self, channel = 1):
        '''Find the 6 hour cadence
        '''
        mov=k2movie.movie(loc=(500,500),
                          channel=[channel],
                          campaign=self.campaign,
                          verbose=False,
                          data_dir=self.database_dir,
                          cadence='sixhour',
                          scale='linear',
                          text=False,vmin=-20,vmax=20,colorbar=True,tol=600,cmap='inferno')
        with silence():
            self.cadence = mov.calc_6hour()



    def build(self,channel):
        '''Grab all the pixels from a channel'''
        try:
            mov=k2movie.movie(loc=(500,500),
                              channel=[channel],
                              campaign=self.campaign,
                              verbose=False,
                              data_dir=self.database_dir,
                              cadence='sixhour',
                              scale='linear',
                              text=False,vmin=-20,vmax=20,colorbar=True,
                              tol=600,cmap='inferno')
            if (self.cadence is None) is False:
                mov.cadence = self.cadence
            else:
                mov.calc_6hour()
            mov.produce()
            mov.populate()
        except:
            return

        dat=np.copy(mov.ar)
        a = np.nanmean(dat,axis=2)
        locs=np.where(np.isfinite(a))
        a = a[np.isfinite(a)].ravel()

        pix = np.zeros((np.shape(locs)[1],np.shape(dat)[2]))
        for i,loc in enumerate(np.transpose(locs)):
            y=dat[loc[0],loc[1]]
            pix[i,:] = y
        self.pix = pix
        self.locs = locs
        self.channel = channel


    def rolling(self,npoly=3):
        '''Create images of rolling band noise and other heating
        '''
        for name, dim in zip(['Column','Row'], [self.locs[0],self.locs[1]]):
            s = np.sort(np.unique(dim))
            mpix=np.zeros((len(s),np.shape(self.pix)[1]))
            x = np.arange(np.shape(self.pix)[1])
            for i,l in enumerate(s):
                y2d = self.pix[dim==l,:]
                y2d-=np.nanmin(y2d)
                y2d+=1
                y2d /= np.atleast_2d(np.nanmedian(y2d,axis=1)).T
                y=np.nanmean(y2d, axis=0)
                mpix[i] = y
            mpix/=np.transpose(np.atleast_2d(np.nanmedian(mpix,axis=1)))
            mpix/=np.atleast_2d(np.nanmedian(mpix,axis=0))

            #Correct out a polynomical
            if npoly > 0:
                cpix = np.copy(mpix)*np.nan
                polys = np.zeros((len(cpix),npoly+1))
                for i,m in enumerate(mpix):
                    x = np.arange(len(m))
                    ok = np.where(np.isfinite(m)&(np.abs(m-1)<4*np.nanstd(m)))[0]
                    if len(ok)==0:
                        continue
                    line = np.polyval(np.polyfit(x[ok],m[ok],npoly),x)
                    ok = np.where(np.isfinite(m)&(np.abs(m-1)<3*np.nanstd(m))&(np.abs(m-line)<5*np.nanstd(m-line)))[0]
                    if len(ok)==0:
                        continue
                    polys[i,:]=np.polyfit(x[ok],m[ok],npoly)
                    line = np.polyval(polys[i,:],x)
                    cpix[i,ok]=m[ok]/line[ok]
            else:
                cpix=mpix

            fig, ax = plt.subplots(1, figsize=(7,7))
            cmap = plt.get_cmap('inferno')
            plt.imshow(cpix.T,vmin=0.9, vmax=1.1, cmap=cmap,origin='bottom')
            cbar=plt.colorbar()
            cbar.set_label('Normalised Flux',fontsize=15)
            plt.xlabel('{} Number'.format(name),fontsize=15)
            plt.ylabel('Cadence',fontsize=15)
            ax.set_yticklabels((ax.get_yticks()*12).astype(int))
            ax.set_aspect(3)
            ax.set_title('Campaign {} Channel {}'.format(self.campaign,self.channel),fontsize=20)
            cdir='{}images/'.format(self.dir)+'c{0:02}/'.format(self.campaign)
            if not os.path.isdir(cdir):
                os.makedirs(cdir)
            img_name = '{}{}_{}.png'.format(cdir,self.channel,name)
            plt.savefig(img_name, dpi=200,bbox_inches='tight')

    def power(self, n=400, r=2000, pmin=0.01, pmax=40.):
        '''Create a power spectrum of all pixels
        '''
        f = np.linspace(1./(pmin*u.day).to(u.min),1./(pmax*u.day).to(u.min),r)
        f = np.sort(f)
        top = np.argsort(np.nanmean(self.pix,axis=1))[::-1][0:n]
        par = np.zeros((n,len(f)))
        x = self.cadence*29.42*u.min
        for i,p in enumerate(self.pix[top,:]):
            ok = np.isfinite(p)
            par[i,:]=LombScargle(x[ok],p[ok],dy=1).power(f)
        fig = plt.figure(figsize=(7,7))
        cmap = plt.get_cmap('viridis')
        X,Y=np.meshgrid(f, np.arange(n))
        plt.contourf((1./X), Y, np.log10(par), cmap=cmap, vmin=-4, vmax=0)
        plt.xscale('log')
        plt.xlabel('Period (Minutes)',fontsize=15)
        plt.ylabel('Pixel Number',fontsize=15)
        plt.title('Campaign {} Channel {}'.format(self.campaign,self.channel),fontsize=20)
        cbar = plt.colorbar()
        cbar.set_label('log$_{10}$(Power)',fontsize=15)
        cdir='{}images/'.format(self.dir)+'c{0:02}/'.format(self.campaign)
        if not os.path.isdir(cdir):
            os.makedirs(cdir)
        img_name = '{}{}_PS.png'.format(cdir,self.channel)
        plt.savefig(img_name, dpi=200,bbox_inches='tight')

def run(campaigns=np.arange(1,8)):
    for campaign in campaigns:
        print('Running Campaign {}'.format(campaign))
        k = krill(campaign)
        k.build_cad()
        for ch in tqdm(np.arange(1,85)):
            with silence():
                k.build(ch)
                k.rolling()
                #k.power()
