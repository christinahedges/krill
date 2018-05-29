import matplotlib.pyplot as plt
import dask.dataframe as dd
import pickle
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from IPython.display import Audio
import logging

log = logging.getLogger('krill ')

DATA_DIR = '/Volumes/cupertino/database/'
WCS_DIR = '/Users/ch/K2/projects/k2movie/k2movie/data/wcs/'


def run():
    campaigns = np.arange(9)
    channels = np.arange(1, 85)
    for campaign in campaigns:
        for ch in channels:
            krill(campaign, ch)


def krill(campaign, ch, overwrite=False, dir='../images/'):
    cdir = '{}'.format(dir)+'c{0:02}/'.format(campaign)
    if not os.path.isdir(cdir):
        os.makedirs(cdir)
    edir = '{}'.format(cdir)+'{0:02}/'.format(ch)
    if not os.path.isdir(edir):
        os.makedirs(edir)
    if (os.path.isfile('{}'.format(edir)+'row_c{0:02}_ch{1:02}.png'.format(campaign, ch))):
        if overwrite == False:
            log.warning(
                'File C{0:02} Ch{1:02} Exists. Set overwrite to True.'.format(campaign, ch))
            return

    fname = '{0}c{1:02}/{2:02}/k2movie_c{1:02}_ch{2:02}.h5'.format(DATA_DIR, campaign, ch)
    if not os.path.isfile(fname):
        log.warning('C{} CH{}: No File'.format(campaign, ch))
        return
    df = dd.read_hdf(fname, 'table')
    # No pixels inside pipeline apertures
    df = df[df.APERFLAG == 1]

    # No cadences where there was only NaNs
    cad = list(df.columns[5:])
    tmean = np.asarray(df[cad].mean(axis=0, skipna=True).compute())
    ok = np.abs(tmean - np.median(tmean[tmean != 0])) < 3 * np.std(tmean[tmean != 0])
    cad = np.asarray(df.columns[5:])[ok]
    cad = list(np.append(np.asarray(['Row', 'Column']), cad))
    df = df[cad]

    df['min'] = df[cad[len(cad)//2-250: len(cad)//2+250]].min(skipna=True, axis=1)
    df['mean'] = df[cad[len(cad)//2-250: len(cad)//2+250]].mean(skipna=True, axis=1)
    df['std'] = df[cad[len(cad)//2-250: len(cad)//2+250]].std(skipna=True, axis=1)

    # Not bright
    minimum = np.percentile(df['min'], 30) + 200
    df = df[(df['mean'] != 0) & (df['mean'] < minimum)]

    # Not flickering
    stdclip = np.percentile(df['std'], 30)
    df = df[(df['std'] <= stdclip)]

    # Group by column and row
    col = df.groupby('Column').mean().compute()
    row = df.groupby('Row').mean().compute()
    col = col[col.columns[1:-3]]
    row = row[row.columns[1:-3]]

    # Add nans back in for missing times
    t = np.asarray(col.columns, dtype=int)
    allt = np.arange(t.min(), t.max())
    missingt = allt[~np.in1d(allt, t)]
    for t in missingt:
        col['{}'.format(t)] = np.nan
        row['{}'.format(t)] = np.nan
    col = col[np.asarray(col.columns)[np.argsort(np.asarray(col.columns))]]
    row = row[np.asarray(col.columns)[np.argsort(np.asarray(col.columns))]]

    # Add nans back in for missing columns
    low = np.asarray(col.index[:-1][(col.index[1:] - col.index[:-1]) > 1] + 1, dtype=int)
    high = np.asarray(col.index[1:][(col.index[1:] - col.index[:-1]) > 1] - 1, dtype=int)
    for l, h in zip(low, high):
        col.loc[l] = np.nan
        col.loc[h] = np.nan
    col = col.sort_index()

    # Add nans back in for missing rows
    low = np.asarray(row.index[:-1][(row.index[1:] - row.index[:-1]) > 1] + 1, dtype=int)
    high = np.asarray(row.index[1:][(row.index[1:] - row.index[:-1]) > 1] - 1, dtype=int)
    for l, h in zip(low, high):
        row.loc[l] = np.nan
        row.loc[h] = np.nan
    row = row.sort_index()

    # Plot both
    for tab, label in zip([col, row], ['Column', 'Row']):
        fig, ax = plt.subplots(figsize=(9, 8))
    #    ax.set_facecolor('black')
        t = np.asarray(tab.columns, dtype=int)
        x = np.asarray(tab.index, dtype=int)
        T, X = np.meshgrid(t, x)
        ar = np.asarray(tab)
        ar -= np.atleast_2d(np.nanmedian(ar, axis=0))
        ar -= np.atleast_2d(np.nanmedian(ar, axis=1)).T
        im = ax.pcolormesh(T, X, ar, vmin=-15, vmax=15, cmap='bwr')
        plt.gca().set_aspect(ar.shape[1]/ar.shape[0])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Background Flux [e$^-$/s]')
        plt.xlabel('Cadence Number')
        plt.ylabel('{} Number'.format(label))
        plt.title('Campaign {} Channel {} ({}-wise)'.format(campaign, ch, label))
        fig.savefig('{}'.format(edir)+'{2}_c{0:02}_ch{1:02}.png'.format(campaign, ch, label))
    return
