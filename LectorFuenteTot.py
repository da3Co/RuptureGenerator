# -*- coding: utf-8 -*-
"""
@author: David Alejandro Castro Cruz
da.castro790@uniandes.edu.co
Organization: KAUST
"""
import numpy as np
import matplotlib.pyplot as plt
from os import path, makedirs, listdir
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from tqdm import tqdm
from scipy import optimize
import pickle
def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return int(2 ** m_i)
def fourSpec(data, frem, tnd=-1):
    if tnd==-1: tnd=nextpow2(data.size)
    FT = np.abs(np.fft.rfft(data, tnd))
    df = frem / tnd
    return [np.arange(FT.size-1)*df, FT/frem]
from scipy.io import savemat
mpl.use('TkAgg')  # interactive mode works with this, pick one
suf='Example2'
fold = 'Results/%s/Sources/'%suf  # RaSN AIS1
folp = 'plots/%s/SourcesAll/'%suf  # RaSN # Folder to save the plots

envio=None#'envio/'
suf='PB'

col = (0, 0.5, 1.0)  # '2008S1',
fss = [2*6.4, 2.4*4.8]  # [4.8, 6.4]
fssT =[6.4, 2.6]
time = np.arange(0, 20, 1)  # dynamic plots times 30
ffb = (0.01, 2.5)  # opt range black dash line

levTo = np.linspace(0, 14, 15)
levS = np.linspace(0, 4.0, 41)
levVpk = np.arange(0, 6.5, 0.1)
levRVr = np.arange(0.1, 1.1, 0.01)
levTr = np.arange(0, 8.1, 0.2)
levTpk = np.linspace(0, 0.3, 31)
levSt = np.linspace(-0.2, 0.2, 21)
#  -------------------------------------------------------------------------------------------------------------
lfil =[file[:-7] for file in listdir(fold) if file.endswith('.pickle')]

p=0.2
colormap = plt.cm.get_cmap('turbo')
colA = np.asarray([colormap(pp) for pp in np.linspace(0, 1, levS.size)])
nnb=int(np.round(colA.shape[0]*p))
colN=1-np.transpose(np.tile(np.arange(nnb)/nnb, (4,1)))*np.tile((1-colA[nnb]), (nnb,1))
cma = np.append(colN, colA[nnb:], axis=0)
cmap = ListedColormap(cma[:,:-1])

figTo = plt.figure()
figFT = plt.figure()

FEsp={}#
[lfil.remove(file) for file in FEsp.keys()]
lfil=lfil+list(FEsp.keys())
for file in tqdm(lfil):
#for nfil in tqdm(range(80)):#
    #file='Si_%d' % nfil
    if not path.exists('%s' % folp): makedirs('%s' % folp[:-1])
    with open(fold + file + '.pickle', 'rb') as fi:
        So = pickle.load(fi)
        fi.close()

        nl, nw = So.Slip.shape
        dt = So.timiT[1] - So.timiT[0]

        stf = np.gradient(So.SeismicMoment, dt)
        # Moment rate
        plt.figure(figsize=fssT)
        plt.subplot(1, 2, 1)
        plt.plot(So.timiT, stf, linewidth=2.5, color=col)
        plt.ylabel('$\\dot{Mo}$ [$Nm/s$]', fontsize=16)
        plt.xlabel('Time [$s$]', fontsize=16)
        plt.xlim(0, time[-1])  # dt*nt)#25)#
        plt.ylim(ymin=0)

        # spectra rate\
        plt.subplot(1, 2, 2)
        freq, FT = fourSpec(np.interp(np.arange(0, 1000, dt), So.timiT, stf, 0, 0), 1 / dt)
        plt.plot(freq[1:], FT[1:-1], linewidth=2.5, color=col)

        ffs = np.where(np.all([freq > ffb[0], freq < ffb[1]], axis=0))[0]
        wff = np.log10(freq[ffs + 1] / freq[ffs])
        C = FT[0]

        ep = lambda fc: np.sum(np.log(C / (1 + freq[ffs] ** 2 / fc ** 2) / FT[ffs]) * wff / (fc ** 2 / freq[ffs]** 2+1))

        # np.savetxt('FTSref.txt', np.transpose([freq, FT[:-1]]), '%g', delimiter=',', header='freq[hz], FT_stf [NM]')
        resu = optimize.fsolve(ep, 10 ** (16 / 3 - np.log10(So.Mo) / 3))
        # resu[0]=0.04
        plt.plot(freq, C / (1 + freq ** 2 / resu[0] ** 2), color=(0, 0, 0), linewidth=2, linestyle='--')

        '''C=np.sum((np.log10(FT[ffs])+2*np.log10(freq[ffs]))*wff)/np.sum(wff)
        #C=np.sum(np.log10(FT[ffs])+2*np.log10(freq[ffs]))
        Y=10**(-2*np.log10(freq)+C)
        plt.plot(freq, Y, color=(0,0,0), linewidth = 2, linestyle = '--')
        '''
        fma = np.min(So.Vs) / (3 * min([So.dww, So.dll]))
        plt.axvline(fma, color=(0, 0, 1), linewidth=1)
        plt.ylabel('|FT| [$Nm$]', fontsize=16)
        plt.xlabel('Frequency [$Hz$]', fontsize=16)
        #plt.xlim(freq[1], 30)
        plt.xlim(1E-2, 20)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1E-5*So.Mo, 2.5*So.Mo)
        plt.grid()
        plt.tight_layout()
        plt.savefig('%sGeneral2%s.png' % (folp, file), dpi=120)#, transparent=True)
        plt.close()

        plt.figure(num=figTo.number)
        if file in FEsp:
            plt.plot(So.timiT, stf, linewidth=2.5, color=FEsp[file][1], label=FEsp[file][0])
        else:        
            plt.plot(So.timiT, stf, linewidth=1.5, color=(0.6,0.6,0.6))


        plt.figure(num=figFT.number)
        if file in FEsp:
            plt.plot(freq[1:], FT[1:-1], linewidth=2.5, color=FEsp[file][1], label=FEsp[file][0])
        else:        
            plt.plot(freq[1:], FT[1:-1], linewidth=1.5, color=(0.6,0.6,0.6))


        plt.figure(figsize=fss)
        # Moment rate points
        dl, dw = So.dll, So.dww
        lvec = 0.001 * np.arange(nl) * dl
        wvec = 0.001 * np.arange(nw) * dw

        #  Splip-to plot
        plt.subplot(3,2,1)
        ax1 = plt.gca()
        plt.contourf(lvec, wvec, np.transpose(So.Slip), cmap=cmap, levels=levS, extend='max')
        #levs = np.arange(0, (np.nanmax(Mt[1:-1, 1:-1])//2)*2+0.1, 2.0)  # np.nanmax(10)

        plt.title('Slip [$m$]', fontsize=14)
        plt.ylabel('Width direction [$km$]', fontsize=16)
        plt.xlabel('Length direction [$km$]', fontsize=16)
        cbar = plt.colorbar(orientation='vertical')
        #cbar.ax.set_title('[$m$]', fontsize=14)  # ----------------------------------------------------------------
        # C = plt.contour(lvec, wvec, np.flip(np.transpose(Mt), axis=1), levels=levs, cmap='winter', linewidths=0.8)
        C = plt.contour(lvec, wvec, np.transpose(So.To), levels=levTo, cmap='gist_gray_r', linewidths=2.0)
        ax1.clabel(C, inline=1, fontsize=8, fmt='%.1f')
        plt.xlim(0, lvec[-1])
        plt.ylim(0, wvec[-1])

        #phy = np.unravel_index(np.nanargmin(Mt[1:-1, 1:-1]), Slp.shape)
        phyC=np.interp(So.hypo[0], np.arange(lvec.size), lvec), np.interp(So.hypo[1], np.arange(wvec.size), wvec)

        plt.scatter(phyC[0], phyC[1], s=40, facecolors='none', edgecolors=(0.940015, 0.975158, 0.131326), marker='*')

        ax1.set_aspect('equal', 'box')

        # Rise time plot
        plt.subplot(3,2,3)
        ax1 = plt.gca()
        trise=So.Trise.copy()
        trise[trise > levTr[-1]] = levTr[-1]
        plt.contourf(lvec, wvec, np.transpose(trise), cmap='jet', levels=levTr, extend='max')  # bonecmap
        cbar = plt.colorbar(orientation='vertical')
        C = plt.contour(lvec, wvec, np.transpose(So.To), levels=levTo, cmap='gist_gray_r', linewidths=2.0)
        #ax1.clabel(C, inline=1, fontsize=8, fmt='%.1f')
        plt.scatter(phyC[0], phyC[1], s=40, facecolors='none', edgecolors=(0.940015, 0.975158, 0.131326), marker='*')

        plt.title('$\\tau_{rise}$ [$s$]', fontsize=14)
        #cbar.ax.set_title('$\\tau_{rise}$ [$s$]', fontsize=14)
        plt.ylabel('Width direction [$km$]', fontsize=16)
        plt.xlabel('Length direction [$km$]', fontsize=16)
        plt.xlim(0, lvec[-1])
        plt.ylim(0, wvec[-1])

        ax1.set_aspect('equal', 'box')

        # Tpeak plot
        Tpeak=So.tac[:]
        plt.subplot(3, 2, 2)
        ax1 = plt.gca()
        Tpeak[Tpeak > levTpk[-1]] = levTpk[-1]
        plt.contourf(lvec, wvec, np.transpose(Tpeak), cmap='jet', levels=levTpk, extend='max')  # bonecmap
        cbar = plt.colorbar(orientation='vertical')
        C = plt.contour(lvec, wvec, np.transpose(So.To), levels=levTo, cmap='gist_gray_r', linewidths=2.0)
        # ax1.clabel(C, inline=1, fontsize=8, fmt='%.1f')
        plt.scatter(phyC[0], phyC[1], s=40, facecolors='none', edgecolors=(0.940015, 0.975158, 0.131326), marker='*')
        plt.title('$Tpeak$ [$s$]', fontsize=14)
        plt.ylabel('Width direction [$km$]', fontsize=16)
        plt.xlabel('Length direction [$km$]', fontsize=16)
        plt.xlim(0, lvec[-1])
        plt.ylim(0, wvec[-1])
        ax1.set_aspect('equal', 'box')

        # Vpeak plot
        Vpeak = So.SlipRateMax[:]
        plt.subplot(3, 2, 4)
        ax1 = plt.gca()

        Vpeak[Vpeak > levVpk[-1]] = levVpk[-1]
        plt.contourf(lvec, wvec, np.transpose(Vpeak), cmap='jet', levels=levVpk, extend='max')  # bonecmap
        cbar = plt.colorbar(orientation='vertical')
        C = plt.contour(lvec, wvec, np.transpose(So.To), levels=levTo, cmap='gist_gray_r', linewidths=2.0)
        # ax1.clabel(C, inline=1, fontsize=8, fmt='%.1f')
        plt.scatter(phyC[0], phyC[1], s=40, facecolors='none', edgecolors=(0.940015, 0.975158, 0.131326), marker='*')
        plt.title('$Vpeak$ [$m/s$]', fontsize=14)
        #cbar.ax.set_title('$Vpeak$ [$m/s$]', fontsize=14)
        plt.ylabel('Width direction [$km$]', fontsize=16)
        plt.xlabel('Length direction [$km$]', fontsize=16)
        plt.xlim(0, lvec[-1])
        plt.ylim(0, wvec[-1])
        # plt.tight_layout()
        ax1.set_aspect('equal', 'box')

        #  prob Hypocenter x,y
        if hasattr(So, 'pHyp'):
            plt.subplot(3, 2, 5)
            Phyp = np.log10(So.pHyp[:])
            val = np.isfinite(Phyp)
            levPhyp = np.linspace(np.max(Phyp[val])-4, np.max(Phyp[val]), 25)
            Phyp[np.logical_not(val)] = levPhyp[0]
            Phyp[Phyp<levPhyp[0]] = levPhyp[0]

            ax1 = plt.gca()
            plt.contourf(lvec, wvec, np.transpose(Phyp), cmap='jet', levels=levPhyp, extend='max')  # bonecmap
            cbar = plt.colorbar(orientation='vertical')
            C = plt.contour(lvec, wvec, np.transpose(So.To), levels=levTo, cmap='gist_gray_r', linewidths=2.0)
            # ax1.clabel(C, inline=1, fontsize=8, fmt='%.1f')
            plt.scatter(phyC[0], phyC[1], s=40, facecolors='none', edgecolors=(0.940015, 0.975158, 0.131326), marker='*')
            plt.title('$P_{hyppcenter}$ [$1$]', fontsize=14)
            plt.ylabel('Width direction [$km$]', fontsize=16)
            plt.xlabel('Length direction [$km$]', fontsize=16)
            plt.xlim(0, lvec[-1])
            plt.ylim(0, wvec[-1])
            ax1.set_aspect('equal', 'box')
        # Vr/Vs plot
        plt.subplot(3, 2, 6)
        ax1 = plt.gca()

        rVr=So.Vr[:]/So.Vs[:]
        rVr[rVr > levRVr[-1]] = levRVr[-1]
        plt.contourf(lvec, wvec, np.transpose(rVr), cmap='jet', levels=levRVr, extend='max')  # bonecmap
        cbar = plt.colorbar(orientation='vertical')
        C = plt.contour(lvec, wvec, np.transpose(So.To), levels=levTo, cmap='gist_gray_r', linewidths=2.0)
        # ax1.clabel(C, inline=1, fontsize=8, fmt='%.1f')
        plt.scatter(phyC[0], phyC[1], s=40, facecolors='none', edgecolors=(0.940015, 0.975158, 0.131326), marker='*')

        plt.title('$V_{rup}$/$Vs$ [$1$]', fontsize=14)
        # cbar.ax.set_title('$\\tau_{rise}$ [$s$]', fontsize=14)
        plt.ylabel('Width direction [$km$]', fontsize=16)
        plt.xlabel('Length direction [$km$]', fontsize=16)
        plt.xlim(0, lvec[-1])
        plt.ylim(0, wvec[-1])

        ax1.set_aspect('equal', 'box')

        plt.tight_layout()
        plt.savefig('%sGeneral%s.png' % (folp, file), dpi=300)#, transparent=True)

        plt.close()

        '''if not(envio is None):
            if not path.exists('%s' % envio): makedirs('%s' % envio[:-1])
            params = {'Slp': Slp, 'hyp_ind': np.asarray(phyC)}
            savemat('%s%s%s.mat'% (envio, suf, file), {'params': params})'''

plt.figure(num=figFT.number)
plt.axvline(fma, color=(0, 0, 1), linewidth=1)
plt.ylabel('|FFT(\\dot{Mo})| [$Nm$]', fontsize=16)
plt.xlabel('Frequency [$Hz$]', fontsize=16)
# plt.xlim(freq[1], 30)
plt.xlim(1E-2, 20)
plt.xscale('log')
plt.yscale('log')
# plt.ylim(1E14, 1E20)
plt.grid()
plt.tight_layout()
plt.savefig('%sGG_FT.png' % folp, dpi=300)  # , transparent=True)

plt.figure(num=figTo.number)
plt.legend(loc=0, fontsize=14)
plt.ylabel('$\\dot{Mo}$ [$Nm/s$]', fontsize=16)
plt.xlabel('Time [$s$]', fontsize=16)
plt.xlim(0, time[-1])  # dt*nt)#25)#
plt.ylim(ymin=0)
plt.tight_layout()
plt.savefig('%sGG_To.png' % folp, dpi=300)  # , transparent=True)
print('%g [Hz]' % fma)
print('Hecho!!')