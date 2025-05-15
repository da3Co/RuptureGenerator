# -*- coding: utf-8 -*-
"""
@author: David Alejandro Castro Cruz
da.castro790@uniandes.edu.co
Insitution: King Abdullah University of Science and Technology
"""
import numpy as np
import RuptG as rg
import matplotlib.pyplot as plt
from os import path, makedirs
from multiprocessing import Pool, cpu_count
import matplotlib as mpl
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator
from functools import partial
from time import time
import h5py
from scipy.optimize import fsolve, minimize
import warnings
import copy
warnings.filterwarnings("ignore")
mpl.use('TkAgg')  # interactive mode works with this, pick one
def addHFSource(sii, LRR, Inp, patt=True):
    '''

    :param sii: Index of the simulation
    :param Inp: Inputs lists
    :param patt: is the main branch
    :return: Creates a rupture process and write it to hdf5 files
    '''

    (Mo, L_I, W_I, stri, dip, rake, loc, dll, dww, tapPL_G, LVumaxI, phaseI, stRa, trp, vpkMax, phaseI_Tr,
        CVrr_I, stdVr, phaseI_Vr, addSSV, sufi) = Inp

    RR = LRR[sii]

    #  Slip pattern definition
    if randUmax: sucS = RR.setSlipPatternHF(loc[:3], phaseI, tapPL_G, None, LVumaxI[sii])

    if sucS:
        #  Rakes variation definition
        RR.setRakesVariations(stRa)

        #  Rise Time definition
        sucTr = RR.setRiseTimePatternHF(loc[6:], KtacM, trp, phaseI_Tr, [0.4, 1], vpkMax)
        if sucTr:
            #  Rupture velocity pattern and define Onset times
            sucV = RR.setRuptVelocHF(CVrr_I, stdVr, loc[3:6], phaseI_Vr, cVpVr=[0.15, 0.7])

            if sucV:
                #  Compute STF at each subfault
                RR.generateSTFOpt(dtt, addSSV)

                #  Write sources
                RR.writePickle(foldS + '%s%d.pickle' % (sufi, sii))

    csus=sucS and sucTr and sucV
    if not(csus) and patt:
        ii=1
        succ=False
        while ii<nmit or not(succ):
            ii+=1
            succ=addHFSource(sii, LRR, Inp, False)
        if not(succ):
            print('Rupture no generated: [%r, %r, %r, %r]' %(sucS, sucTr, sucV, hasattr(RR, 'To')))

    return csus

def CSuelo(zz):
    Da = np.loadtxt('geo/1DLA.txt', skiprows=1, delimiter=',')
    #  mechanical Properties  ------------------------------------------------------------------------------------
    Vs = Da[0][2] * np.ones(zz.shape)
    Vp = Da[0][1] * np.ones(zz.shape)
    Rho = Da[0][3] * np.ones(zz.shape)

    Qs = Da[0][5] * np.ones(zz.shape)
    Qk = Da[0][4] * np.ones(zz.shape)
    for dla in Da[1:]:
        re=zz<-dla[0]
        Vs[re] = dla[2]
        Vp[re] = dla[1]
        Rho[re] = dla[3]

        Qs[re] = dla[5]
        Qk[re] = dla[4]
    return Vp, Vs, Rho
if __name__ == '__main__':
    nreal = 10  # Number of realizations
    foldS = 'Results/Example3/'  # Output folder
    dll, dww = 2.0 * 100.0, 2.0 * 100.0  #  Grid spacing in the fault
    dtt = 1E-3  # Step time
    fileE = 'CulS0/CulS0'  # Initial Source
    tapPL_G = np.asarray([0.15, 0.15, 0.15, 0.0])  # Tapper definition [-x,+x,-y,+y]
    fLim = 0.4  # Limit frequency of the Old source
    umaxP=1.2# factor to compute umax from the inversion (umax=umaxP*umax_inv)
    randUmax = True  # Random selection of the maximal slip
    addSSV = True  # Add small scale variations
    sufi = 'Si_'  #  Suffix names of the sources
    KtacM = 0.08  # Tpeak average
    # ----------------------------------------------  Fixed inputs
    H = 0.77  # Mai and Beroza, 2002
    Hvr = -0.3
    N = 4  # Sharpness transition parameter
    stRa = np.deg2rad(15)  # --radians-- standard deviation of rake variations (Graves Pitarka, 2010)
    vpkMax = 6.5  # Maximal slip rate
    #  ----------------------------------------------------------------------------------------------  End inputs
    nmit = 25  # max number of iterations to solve a source
    if not path.exists('%s' % foldS): makedirs(foldS[:-1])
    ncore = cpu_count()

    fe = h5py.File(fileE + '.hdf5', 'r')
    #  Dimensions extraction
    L_I = fe.attrs['LL']  # in meters
    W_I = fe.attrs['LW']  # in meters

    stri = np.rad2deg(fe.attrs['strike'])  # Strike
    dip = np.rad2deg(fe.attrs['dip'])  # Dip
    rake = np.rad2deg(np.mean(fe['rake']))  # Average Rake

    PoiL = fe.attrs['Hypo'][:]  # Absolute location of PoiR
    hypoP = (fe.attrs['dl'] * (0.5+fe.attrs['Hy_dnLdnW'][0]), fe.attrs['dw'] * (0.5+fe.attrs['Hy_dnLdnW'][1]))

    nll0 = np.concatenate(([-0.5 * fe.attrs['dl']], np.arange(fe['Mom'].shape[1]) * fe.attrs['dl'],
                           [(fe['Mom'].shape[1] - 0.5) * fe.attrs['dl']]))+0.5*fe.attrs['dl']
    nww0 = np.concatenate(([-0.5 * fe.attrs['dw']], np.arange(fe['Mom'].shape[2]) * fe.attrs['dw'],
                           [(fe['Mom'].shape[2] - 0.5) * fe.attrs['dw']]))+0.5*fe.attrs['dw']

    #  Generation of the new fault
    Mo = fe.attrs['Mo']
    Mw = (np.log10(Mo) - 9.05) / 1.5
    RR = rg.Rupture(Mo, L_I, W_I)

    PoiR_I=[hypoP[0]/L_I, hypoP[1]/W_I]

    RR.assingLocationsOrientations(stri, dip, rake, dll, dww, PoiR_I, PoiL)

    _, Vs, Rho = CSuelo(RR.Pos[2])
    RR.assingMaterialProp(Vs, Rho)

    #  Definition of the weigth function
    RR.defineWeigthFunction(fLim, N)

    #  set Hypocenter
    chypo = int(np.round(hypoP[0]/RR.dll)), int(np.round(hypoP[1]/RR.dww))
    RR.setHypocenter(None, None, chypo)

    #  Estimation correlations lengths
    Mof0 = np.zeros((nll0.size, nww0.size))
    Mof0[1:-1, 1:-1] = np.sqrt(np.sum(fe['Mom'][:, :, :, -1] ** 2, axis=0))
    interSlp = RegularGridInterpolator((nll0, nww0), Mof0, method='linear', bounds_error=False, fill_value=0.0)
    SlpP = np.transpose(interSlp(np.transpose([RR.Dll, RR.Dww])))
    kw = np.fft.rfftfreq(SlpP.shape[1], dww)
    kl = np.fft.fftfreq(SlpP.shape[0], dll)
    Dks, Dkd = np.meshgrid(kl, kw, indexing='ij')
    D = np.fft.rfft2(SlpP)

    He = H + 1

    fu = lambda lof: np.sum(RR.F * np.log(np.abs(D[0, 0]) *
                    np.sqrt(1 / (1 + (lof[0] * Dks) ** 2 + (lof[0] * lof[1] * Dkd) ** 2) ** He) / np.abs(D)) ** 2)
    cll = 2 * np.pi * np.mean([10 ** (-2.928 + 0.588 * Mw), 1.855 + 0.341 * L_I * 1E-3,
                               -4.870 + 0.741 * W_I * 1E-3], axis=0) * 1E3

    res = minimize(fu, np.asarray([cll, 1.0]), bounds=((0.1 * fe.attrs['LL'], 10 * fe.attrs['LL']), (0.1, 1)))

    cll_To, cww_To = res.x[0], res.x[0] * res.x[1]
    cll_Tr, cww_Tr = res.x[0], res.x[0] * res.x[1]
    loc = np.asarray([res.x[0], res.x[0] * res.x[1], He, cll_To, cww_To, Hvr + 1, cll_Tr, cww_Tr, He])

    #  Original phases computations
    FT2 = np.fft.rfft2(SlpP)
    theta0 = np.angle(FT2)
    theta0[0, 0] = 0.0
    phaseSP = np.exp(theta0 * 1j)

    interTr = RegularGridInterpolator((nll0[1:-1], nww0[1:-1]), fe['trise'][:], method='linear', bounds_error=False)
    trrP = interTr(np.transpose([RR.Dll, RR.Dww])).T
    co = np.isfinite(trrP)
    interTr = NearestNDInterpolator((RR.Dll[co], RR.Dww[co]), trrP[co])
    fa = np.logical_not(co)
    trrP[fa] = interTr(np.transpose([RR.Dll[fa], RR.Dww[fa]]))

    utr = trrP ** 2
    utr *= Mo / np.sum(dll * dww * utr * Rho * Vs ** 2)

    theta0 = np.angle(np.fft.rfft2(utr))
    theta0[0, 0] = 0.0
    phaseI_Tr = np.exp(theta0 * 1j)

    interVr = RegularGridInterpolator((nll0[1:-1], nww0[1:-1]), fe['Vrup'][:] / fe['Vss'][:],
                                     method='linear', bounds_error=False)
    VraP = interVr(np.transpose([RR.Dll, RR.Dww])).T
    co = np.isfinite(VraP)
    interVr = NearestNDInterpolator((RR.Dll[co], RR.Dww[co]), VraP[co])
    fa = np.logical_not(co)
    VraP[fa] = interVr(np.transpose([RR.Dll[fa], RR.Dww[fa]]))
    CVrr_I = np.mean(VraP)
    stdVr = np.std(VraP)

    theta0 = np.angle(np.fft.rfft2(1 - VraP))
    theta0[0, 0] = 0.0
    phaseI_Vr = np.exp(theta0 * 1j)

    trp_I = np.nanmean(fe['trise'])
    fe.close()


    if randUmax: LVumaxI = umaxP*10**(0.1 * np.random.randn(nreal))
    else: LVumaxI = umaxP*np.ones(nreal)

    Inp = (Mo, L_I, W_I, stri, dip, rake, loc, dll, dww, tapPL_G, LVumaxI, phaseSP, stRa, trp_I, vpkMax, phaseI_Tr,
           CVrr_I, stdVr, phaseI_Vr, addSSV, sufi)

    #addHFSource(0, [RR]*nreal, Inp)

    tic = time()
    func = partial(addHFSource, LRR=[copy.deepcopy(RR) for sii in range(nreal)], Inp=Inp, patt=True)
    with Pool(ncore) as pool:
        list(pool.map(func, np.arange(0, nreal)))  # pool.#range(niter)

        pool.close()
    print('%f [s]' % (time() - tic))
    print('Fin :)')
    print('Fin!!!')
