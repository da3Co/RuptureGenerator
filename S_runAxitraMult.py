import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import h5py
import shelve
import pickle
from os import path, makedirs, listdir, remove, devnull
import sys
from multiprocessing import Pool, cpu_count
import shutil
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')  # interactive mode works with this, pick one
#foAx='/project/k10044/David/Programs/axitra-master/MOMENT_DISP_F90_OPENMP/src'
foAx = '/home/castda0a/Programs/axitra-master/MOMENT_DISP_F90_OPENMP/src'

sys.path.append(foAx)
from axitra import *

def Convolove(nfi):
    idd = nfi + ido
    #sys.stdout = open(devnull, 'w')
    if nfi != 0:
        apf = copy.deepcopy(apO)
        apf.sid = 'axi_%d' % idd
        apf.id = idd

        shutil.copy('axi_1.data', 'axi_%d.data' % idd)
        shutil.copy('axi_1.head', 'axi_%d.head' % idd)
        shutil.copy('axi_1.res', 'axi_%d.res' % idd)
        shutil.copy('axi_1.source', 'axi_%d.source' % idd)
        shutil.copy('axi_1.stat', 'axi_%d.stat' % idd)
    else:
        apf = copy.copy(apO)
    #  STF
    cstf = np.interp(tt, tii, cum[nfi], 0, cum[nfi][-1])
    cstf = tt[1] * cstf / cstf[-1]
    # source index, moment, strike, dip, rake,0.,0.,time_delay
    hist = np.array([[1, cum[nfi][-1], strike, dip, np.degrees(rake[0]), 0.0, 0.0, 0.0]])
    # Run
    _, ayP, axP, azP = moment.conv(apf, hist, source_type=3, unit=3, t0=1.0, sfunc=cstf)
    if nfi != 0:
        remove('axi_%d.data' % idd)
        remove('axi_%d.head' % idd)
        remove('axi_%d.res' % idd)
        remove('axi_%d.source' % idd)
        remove('axi_%d.stat' % idd)

    remove('axi_%d.hist' % idd)
    remove('axi_%d.sou' % idd)
    #sys.stdout = sys.__stdout__

    return ayP[:, :int(tt.size // 2)], axP[:, :int(tt.size // 2)], azP[:, :int(tt.size // 2)]

if __name__ == '__main__':
    #  Optimized for sources with equal geometry (idem)
    dur = 50.0
    lrecC = np.loadtxt('map/StatsCG.txt', skiprows=1, delimiter=',')
    Spp = np.loadtxt('geo/1DLA.txt', skiprows=1, delimiter=',')
    top = 0.0
    fli=2  # lim max runs
    fmax = 5.25
    fil = (fmax, 4)
    # folSsL = 'Sources/TAS%d/'%ide
    folSsL = 'Sources/RS65B/'  # Folder where Sources are
    fols = 'Data/Data_RS65B/'  # Output folder
    campo = 2.5  # Hanning window in the tail
    sufS = 'sta_%d'
    fcon = 'Sdone.txt'  # File to store axitra information
    idech = 'ideSH.txt'  # File to store axitra information
    nsst = 1  # Soure to compute using the same GF (>1 if the sources have the same geometry)
    ncore = cpu_count()
    print(ncore)
    #  -----------------------------------------------------------------------------------------------------------------
    fileSsL = listdir(folSsL)
    # index, lat, lon, depth
    stations = np.array([[nsta, dst[1], dst[0], 0.0 - dst[2]] for nsta, dst in enumerate(lrecC)])

    # thickness (or top), Vp, Vs, rho, Qp, Qs
    model = np.array([lina for lina in Spp])
    if not path.exists(fols[:-1]): makedirs(fols[:-1])
    
    socal = np.random.choice(np.arange(len(fileSsL)), len(fileSsL), False)
    nfd = len(socal) // nsst
    if len(socal) % nsst!=0: nfd+= 1
    nsou = -1

    if path.exists(idech):
        fs = open(idech, 'r')
        ideL = fs.readline()
        fs.close()
        if len(ideL) > 0:
            ideL = ideL[:-1].split(',')
        else:
            ideL = []
    else:
        ideL = []
    ido = 1
    while str(ido) in ideL:
        ido += 1
    fs = open(idech, 'a')
    for ii in range(nsst):
        fs.write('%d,' % (ido+ii))
    fs.close()

    sources = np.array([[1, 0.0, 0.0, 0.0]])
    # Set up model
    apO = Axitra(model, stations, sources, fmax=fmax * 1.4, duration=2 * dur, xl=0., latlon=False,
                 axpath=foAx, id=ido)
    nsd=-1
    for ngg in tqdm(range(nfd)):
        if path.exists(fcon):
            fs = open(fcon, 'r')
            lSou = [lisou[:-1] for lisou in fs.readlines()]
            fs.close()
        else:
            lSou = []
        Lisou = []
        So = []
        while len(Lisou) < nsst and nsou < socal.size-1 and nsd<fli:
            nsou += 1
            fileSs = fileSsL[socal[nsou]]
            suf = fileSs[:-7]
            if not (suf in lSou):
                fs = open(fcon, 'a')
                fs.write(suf + '\n')
                fs.close()
                Lisou.append(suf)
                #  Source
                with open(folSsL + fileSs, 'rb') as fi:
                    So.append(pickle.load(fi))
                    fi.close()

        # print(Lisou)
        # print([So[nfi]['Mom'][0].shape for nfi in range(len(Lisou))])
        tii = So[-1].timiT.copy()
        sx = np.zeros((len(Lisou), len(stations), int(apO.npt // 2)))
        sy = np.zeros((len(Lisou), len(stations), int(apO.npt // 2)))
        sz = np.zeros((len(Lisou), len(stations), int(apO.npt // 2)))

        nuns = So[0].Slip.shape[0]
        nund = So[0].Slip.shape[1]
        Pos = So[0].Pos[:]
        strike = So[0].strike
        dip = So[0].dip
        for idde in range(1):  # So[0]['Mom'].shape[0]
            for ndi in range(nund):  # So[0]['trise'].shape[1])):#for ns in tqdm(range(So['trise'].shape[0])):#
                axl = np.zeros((len(Lisou), len(stations), int(apO.npt // 2)))
                ayl = np.zeros((len(Lisou), len(stations), int(apO.npt // 2)))
                azl = np.zeros((len(Lisou), len(stations), int(apO.npt // 2)))

                for nsi in range(nuns):  # for nd in tqdm(range(So['trise'].shape[1])):#
                    # print(len(So))
                    # print('nsi: %d' %nsi)
                    # print('ndi: %d' %ndi)

                    if sum([So[0].LocalMo[nsi][ndi] is None for nfi in range(len(Lisou))]) == 0.0:# !!!!!!!!!!!!! Liso=1
                        cum = [np.interp(np.arange(tii.size), np.arange(So[nfi].LocalMo[nsi][ndi][0][0],
                            So[nfi].LocalMo[nsi][ndi][0][1]),So[nfi].LocalMo[nsi][ndi][1]) for nfi in range(len(Lisou))]
                        # index, lat, lon, depth
                        sources = np.array([[1, Pos[1, nsi, ndi], Pos[0, nsi, ndi], top - Pos[2, nsi, ndi]]])
                        #  EGF
                        apO = Axitra(model, stations, sources, fmax=fmax * 1.4, duration=2 * dur, xl=0.,
                                    latlon=False, axpath=foAx, id=ido)
                        apO = moment.green(apO)
                        tt = np.linspace(0, 2.0 * dur, apO.npt, endpoint=False)

                        rake = [So[nfi].rakes[nsi, ndi] for nfi in range(len(Lisou))]

                        lnfi=[nfi for nfi in range(len(Lisou)) if cum[nfi][-1] != 0]


                        with Pool(ncore) as pool:
                            resu = np.asarray(list(pool.map(Convolove, lnfi)))#pool.

                        remove('axi_%d.data' % ido)
                        remove('axi_%d.head' % ido)
                        remove('axi_%d.res' % ido)
                        remove('axi_%d.source' % ido)
                        remove('axi_%d.stat' % ido)

                        axl += resu[:, 1]
                        ayl += resu[:, 0]
                        azl += resu[:, 2]

                sx += axl
                sy += ayl
                sz += azl
        # print(ngg)
        t = tt[:int(tt.size // 2)]
        dt = t[1] - t[0]

        #  Saving
        # sx = np.gradient(np.gradient(dx, dt, axis=1), dt, axis=1)
        # sy = np.gradient(np.gradient(dy, dt, axis=1), dt, axis=1)
        # sz = np.gradient(np.gradient(dz, dt, axis=1), dt, axis=1)

        [fp1, fp2] = butter(fil[1], 2 * fil[0] * dt, btype='lowpass')

        cdd = int(campo / t[1])
        ws = np.hanning(2 * cdd)

        for nfi in range(len(Lisou)):
            sx[nfi, :, -cdd:] *= ws[-cdd:]
            sy[nfi, :, -cdd:] *= ws[-cdd:]
            sz[nfi, :, -cdd:] *= ws[-cdd:]

            sx[nfi] = filtfilt(fp1, fp2, sx[nfi], axis=1)
            sy[nfi] = filtfilt(fp1, fp2, sy[nfi], axis=1)
            sz[nfi] = filtfilt(fp1, fp2, sz[nfi], axis=1)

            vx = np.cumsum(sx[nfi], axis=1) * dt
            vy = np.cumsum(sy[nfi], axis=1) * dt
            vz = np.cumsum(sz[nfi], axis=1) * dt

            fsals = fols + Lisou[nfi]  # Output
            nsd+=1
            with shelve.open(fsals, 'n') as al:
                for nss in range(sx.shape[1]):
                    al[sufS % nss] = np.asarray([sx[nfi][nss], sy[nfi][nss], sz[nfi][nss]])
                    al[sufS % nss + '_vel'] = np.asarray([vx[nss], vy[nss], vz[nss]])
                    al[sufS % nss + '_Fdis'] = np.asarray(
                        [np.sum(vx[nss]) * dt, np.sum(vy[nss]) * dt, np.sum(vz[nss]) * dt])

                    # al[sufS % nss + '_dis'] = np.asarray([dx[nss], dy[nss], dz[nss]])
                al['pos'] = lrecC
                al['suf'] = sufS
                al['nsta'] = sx.shape[1]
                al['stats'] = [sufS % nss for nss in range(sx.shape[1])]
                al['time'] = t
                al.close()

            So[nfi].close()

    print('Fin!!')
