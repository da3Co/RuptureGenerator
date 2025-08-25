# -*- coding: utf-8 -*-
"""
@author: David Alejandro Castro Cruz
da.castro790@uniandes.edu.co
Organization: KAUST
"""
import numpy as np
import matplotlib.pyplot as plt
from os import path, makedirs, listdir
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
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
suf='Example1'
fold = 'Results/%s/Sources/'%suf#'Sources/CulS2NN/'#
folp = 'plots/%s/Sources/'%suf# # Folder to save the plots   1_12
envio=None#'envio/'
fss = [6.4, 4.8]  # [4.8, 6.4]
time = np.arange(0, 20, 1)  # dynamic plots times 30
ffb = (0.01, 2.5)  # opt range black dash line
levTo = np.linspace(0, 20, 11)
levS = np.arange(0, 6.1, 0.25)
levVpk = np.arange(0, 6.5, 0.1)
levRVr = np.arange(0.6, 1.01, 0.025)
levTr = np.arange(0, 4.5, 0.25)
levTpk = np.linspace(0, 0.3, 31)
levSt = np.linspace(-0.2, 0.2, 21)

ars=[15,10] ## delta arrows[32,22] #

lpoi = [(0.32, 0.87), (0.55, 0.7), (0.875, 0.76), (0.4, 0.5), (0.72, 0.48), (0.25, 0.10), (0.55, 0.25), (0.85, 0.3)]
#  -------------------------------------------------------------------------------------------------------------
# cmap
p=0.2
colormap = plt.cm.get_cmap('turbo')
colA = np.asarray([colormap(pp) for pp in np.linspace(0, 1, levS.size)])
nnb=int(np.round(colA.shape[0]*p))
colN=1-np.transpose(np.tile(np.arange(nnb)/nnb, (4,1)))*np.tile((1-colA[nnb]), (nnb,1))
cma = np.append(colN, colA[nnb:], axis=0)
cmap = ListedColormap(cma[:,:-1])

#lcol=[(241/256, 143/256, 0/256), (0/256, 166/256, 170/256), (156/256, 111/256, 174/256)]
CSo=(0.5, 0.25, 0)#[244/255, 204/255, 140/255]
lcol=[[0.5, 0.25, 0], [1, 0.5, 0], [0.4, 0, 0.8], [0, 0.6, 0]]
lfil =[file[:-7] for file in listdir(fold) if file.endswith('.pickle')]
#lfil = ['Si_0']
lnam=['Rupture 1', 'Rupture 2', 'Rupture 3']
figMo = plt.figure()
figFMo = plt.figure()
nff=-1
for file in tqdm(lfil):
    nff+=1
    col=lcol[nff]
    if not path.exists('%s' % folp): makedirs('%s' % folp[:-1])

    with open(fold + file + '.pickle', 'rb') as fi:
        So = pickle.load(fi)
        fi.close()
    nl, nw = So.Slip.shape
    dt = So.timiT[1]-So.timiT[0]
    nto = nw * nl

    stf = np.gradient(So.SeismicMoment, dt)
    #  plot Mo rate
    plt.figure(num=figMo.number)
    plt.plot(So.timiT, stf, color=col, linewidth=2.2, label=lnam[nff])

    #  plot FFT Mo rate
    plt.figure(num=figFMo.number)
    freq, FT = fourSpec(np.interp(np.arange(0, 1000, dt), So.timiT, stf, 0, 0), 1 / dt)
    plt.plot(freq[1:], FT[1:-1], linewidth=2.5, color=col)

    dl, dw = So.dll, So.dww
    lvec = 0.001 * np.arange(nl) * dl
    wvec = 0.001 * np.arange(nw) * dw
    Dll, Dww = np.meshgrid(lvec, wvec, indexing='ij')

    #  Slip-to plot
    print(lvec[-1])
    print(wvec[-1])
    fig = plt.figure(figsize=fss)
    plt.clf()
    ax1 = plt.gca()

    plt.contourf(lvec, wvec, np.transpose(So.Slip), cmap=cmap, levels=levS, extend='max')
    plt.ylabel('Dip direction [$km$]', fontsize=16)
    plt.xlabel('Strike direction [$km$]', fontsize=16)
    cbar = plt.colorbar(orientation='horizontal', location='bottom')  # 'vertical')
    cbar.ax.set_title('Slip [$m$]', fontsize=14, y=-2.1)
    C = plt.contour(lvec, wvec, np.transpose(So.To), levels=levTo, cmap='gist_gray', linewidths=1.4)
    ax1.clabel(C, inline=1, fontsize=12, fmt='%.1f')
    plt.xlim(0, lvec[-1])

    U, V = np.cos(So.rakes), np.sin(So.rakes)
    sel = np.zeros(Dll.shape, dtype=bool)
    sel1 = sel.copy()

    sx = int(((lvec.size - 1) % ars[0]) / 2)  # int((lvec.size%ars[0])/2)
    sel[np.arange(sx, lvec.size, ars[0]), :] = True
    sx = int(((wvec.size - 1) % ars[1]) / 2)  # int(wvec.size % ars[1] / 2)
    sel1[:, np.arange(sx, wvec.size, ars[1])] = True
    sel = np.all([sel, sel1], axis=0)
    plt.quiver(Dll[sel].flatten(), Dww[sel].flatten(), U[sel].flatten(), V[sel].flatten(),
               color=(0.4, 0.4, 0.4), scale=1.0, scale_units='xy', units='xy')

    phy = np.unravel_index(np.nanargmin(So.To), So.Slip.shape)
    plt.scatter(lvec[phy[0]], wvec[phy[1]], s=140, facecolors='none',
                edgecolors=(0.0, 0.0, 0.0), marker='*')  # (0.940015, 0.975158, 0.131326)
    plt.scatter(lvec[phy[0]], wvec[phy[1]], s=20, color=(0.0, 0.0, 0.0), marker='.')

    ax1.set_aspect('equal', 'datalim')


    ax1.set_ylim(0, wvec[-1])  # So.attrs['LW'] / 1E3
    '''ax2 = ax1.twinx()
    ax2.set_ylabel('Depth [Km]', fontsize=16)

    dmi = 1E-3 * np.min(So.Pos[2])
    m = (1E-3 * np.max(So.Pos[2]) - dmi) / wvec[-1]
    ax2.set_ylim(ax1.viewLim.bounds[1] * m + dmi, ax1.viewLim.bounds[3] * m + dmi)'''
    # plt.tight_layout()
    plt.savefig('%sSlip%s.png' % (folp, file), dpi=300)  # _Slip
    plt.close()

plt.figure(num=figMo.number)
plt.ylabel('$\\dot{Mo}$ [$Nm/s$]', fontsize=16)
plt.xlabel('Time [$s$]', fontsize=16)
plt.xlim(0, 20)#time[-1])  # dt*nt)#25)#
plt.ylim(ymin=0)
plt.legend(loc=1, fontsize=14)
plt.tight_layout()
plt.savefig('%sMoTime.png' % folp, dpi=300)  # _Slip

plt.figure(num=figFMo.number)

ffs = np.where(np.all([freq > ffb[0], freq < ffb[1]], axis=0))[0]
wff = np.log10(freq[ffs + 1] / freq[ffs])
C = FT[0]
ep = lambda fc: np.sum(
    np.log(C / (1 + freq[ffs] ** 2 / fc ** 2) / FT[ffs]) * wff / (fc ** 2 / freq[ffs] ** 2 + 1))
# np.savetxt('FTSref.txt', np.transpose([freq, FT[:-1]]), '%g', delimiter=',', header='freq[hz], FT_stf [NM]')
resu = optimize.fsolve(ep, 10 ** (16 / 3 - np.log10(So.Mo) / 3))
plt.plot(freq, C / (1 + freq ** 2 / resu[0] ** 2), color=(0, 0, 0), linewidth=2, linestyle='--')
fma = np.min(So.Vs) / (3 * min([dw, dl]))

plt.axvspan(fma, freq[-1], facecolor=(0.7, 0.7, 0.7), alpha=0.8)
#plt.axvline(fma, color=(0.2, 0.2, 0.2), linewidth=1, linestyle='--')
plt.ylabel('|$FFT(\\dot{Mo})$| [$Nm$]', fontsize=16)
plt.xlabel('Frequency [$Hz$]', fontsize=16)
# plt.xlim(freq[1], 30)
plt.xlim(1E-2, 20)
plt.ylim(1E13, 1.2*C)
plt.xscale('log')
plt.yscale('log')
# plt.ylim(1E14, 1E20)
plt.grid()
plt.tight_layout()
plt.savefig('%sMoFFT.png' % folp, dpi=300)  # _Slip
print('Hecho!!')