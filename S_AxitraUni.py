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
    dur = 100.0  #  Duration of the simulation
    lrecC = np.loadtxt('map/Stats.txt', skiprows=1, delimiter=',')  # File with the stations
    Spp = np.loadtxt('geo/1DLA.txt', skiprows=1, delimiter=',')  # Geology 1D Layer
    top = 0.0  # Surface level
    fmax = 2.5  # Maximal frequency
    folSsL = 'Results/Example2/Sources/'  # Folder where Sources are
    fols = 'Results/Example2/Records/'  # Output folder
    campo = 2.5  # Hanning window in the tail
    sufS = 'sta_%d'  # Suffix stations
    fcon = 'Sdone.txt'  # File to store axitra information
    idech = 'ideSH.txt'  # File to run information
    ncore = cpu_count()
    print(ncore)
    #  -----------------------------------------------------------------------------------------------------------------
    fil = (fmax, 4)
    fileSsL = listdir(folSsL)
    # index, lat, lon, depth
    stations = np.array([[nsta, dst[1], dst[0], 0.0 - dst[2]] for nsta, dst in enumerate(lrecC)])

    # thickness (or top), Vp, Vs, rho, Qp, Qs
    model = np.array([lina for lina in Spp])
    if not path.exists(fols[:-1]): makedirs(fols[:-1])
    
    socal = np.random.choice(np.arange(len(fileSsL)), len(fileSsL), False)

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
    fs.write('%d,' % ido)
    fs.close()

    sources = np.array([[1, 0.0, 0.0, 0.0]])
    # Set up model
    apO = Axitra(model, stations, sources, fmax=fmax * 1.4, duration=2 * dur, xl=0., latlon=False,
                 axpath=foAx, id=ido)
    nsd=-1

    for nsou in range(socal.size):
        fileSs = fileSsL[socal[nsou]]
        suf = fileSs[:-7]

        if path.exists(fcon):
            fs = open(fcon, 'r')
            lSou = [lisou[:-1] for lisou in fs.readlines()]
            fs.close()
        else:
            lSou = []
        if not (suf in lSou):
            fs = open(fcon, 'a')
            fs.write(suf + '\n')
            fs.close()
            #  Source
            with open(folSsL + fileSs, 'rb') as fi:
                So=pickle.load(fi)
                fi.close()

            tii = So.timiT.copy()
            sx = np.zeros((len(stations), int(apO.npt // 2)))
            sy = np.zeros((len(stations), int(apO.npt // 2)))
            sz = np.zeros((len(stations), int(apO.npt // 2)))

            nuns = So.Slip.shape[0]
            nund = So.Slip.shape[1]
            Pos = So.Pos[:]
            strike = So.strike
            dip = So.dip

            for ndi in tqdm(range(nund)):  # So[0]['trise'].shape[1])):#for ns in tqdm(range(So['trise'].shape[0])):#
                axl = np.zeros((len(stations), int(apO.npt // 2)))
                ayl = np.zeros((len(stations), int(apO.npt // 2)))
                azl = np.zeros((len(stations), int(apO.npt // 2)))

                for nsi in range(nuns):
                    if not(So.LocalMo[nsi][ndi] is None) and not(So.LocalMo[nsi][ndi][0][0]*tii[1] > dur - 0.1):
                        cum = np.interp(np.arange(tii.size), np.arange(So.LocalMo[nsi][ndi][0][0],
                             So.LocalMo[nsi][ndi][0][1]), So.LocalMo[nsi][ndi][1])
                        # index, lat, lon, depth
                        sources = np.array([[1, Pos[1, nsi, ndi], Pos[0, nsi, ndi], top - Pos[2, nsi, ndi]]])
                        #  EGF
                        apO = Axitra(model, stations, sources, fmax=fmax * 1.4, duration=2 * dur, xl=0.,
                                    latlon=False, axpath=foAx, id=ido)
                        apO = moment.green(apO)
                        tt = np.linspace(0, 2.0 * dur, apO.npt, endpoint=False)

                        rake = So.rakes[nsi, ndi]
                        apf = copy.copy(apO)

                        #  STF
                        cstf = np.interp(tt, tii, cum, 0, cum[-1])
                        cstf = tt[1] * cstf / cstf[-1]
                        # source index, moment, strike, dip, rake,0.,0.,time_delay
                        hist = np.array([[1, cum[-1], strike, dip, np.degrees(rake), 0.0, 0.0, 0.0]])
                        # Run
                        _, ayP, axP, azP = moment.conv(apf, hist, source_type=3, unit=3, t0=1.0, sfunc=cstf)

                        remove('axi_%d.hist' % ido)
                        remove('axi_%d.sou' % ido)
                        # sys.stdout = sys.__stdout__

                        axl += axP[:, :int(tt.size // 2)]
                        ayl += ayP[:, :int(tt.size // 2)]
                        azl += azP[:, :int(tt.size // 2)]

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


            sx[:, -cdd:] *= ws[-cdd:]
            sy[:, -cdd:] *= ws[-cdd:]
            sz[:, -cdd:] *= ws[-cdd:]

            sx = filtfilt(fp1, fp2, sx, axis=1)
            sy = filtfilt(fp1, fp2, sy, axis=1)
            sz = filtfilt(fp1, fp2, sz, axis=1)

            vx = np.cumsum(sx, axis=1) * dt
            vy = np.cumsum(sy, axis=1) * dt
            vz = np.cumsum(sz, axis=1) * dt

            fsals = fols + suf  # Output
            nsd+=1
            with shelve.open(fsals, 'n') as al:
                for nss in range(sx.shape[0]):
                    al[sufS % nss] = np.asarray([sx[nss], sy[nss], sz[nss]])
                    al[sufS % nss + '_vel'] = np.asarray([vx[nss], vy[nss], vz[nss]])
                    al[sufS % nss + '_Fdis'] = np.asarray([np.sum(vx[nss]) * dt, np.sum(vy[nss]) * dt, np.sum(vz[nss])*dt])

                    # al[sufS % nss + '_dis'] = np.asarray([dx[nss], dy[nss], dz[nss]])
                al['pos'] = lrecC
                al['suf'] = sufS
                al['nsta'] = sx.shape[0]
                al['stats'] = [sufS % nss for nss in range(sx.shape[0])]
                al['time'] = t
                al.close()


    print('Fin!!')
