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
from functools import partial
from time import time
import warnings
warnings.filterwarnings("ignore")
mpl.use('TkAgg')  # interactive mode works with this, pick one
def computeSource(sii, patt=True):
    '''

    :param sii: Index of the simulation
    :param patt: is the main branch
    :return: Creates a rupture process and write it to hdf5 files
    '''

    np.random.seed(np.random.randint(1, 10E3) * (1 + sii))
    RR = rg.Rupture(Mo, LI[sii], WI[sii])

    RR.assingLocationsOrientations(np.radians(Lstri[sii]), np.radians(Ldip[sii]), np.radians(Lrake[sii]),
                                   dll, dww, PoiR_I, PoiL)
    sucS, sucTr, sucV = False, False, False

    _, Vs, Rho = CSuelo(RR.Pos[2])
    RR.assingMaterialProp(Vs, Rho)

    loc = [Lcll[sii], Lcww[sii], 1 + H]
    #  Slip pattern definition
    if randUmax: sucS = RR.setSlipPattern(loc, tapPL_G, None, LVumaxI[sii])
    # else: RR.setSlipPattern(loc, tapPL_G,umax )

    if sucS:
        #  Rakes variation definition
        RR.setRakesVariations(stRa)

        #  Rise Time definition
        sucTr = RR.setRiseTimePattern(loc, KtacM, LtrpI[sii], cUTr, vpkMax)

        if sucTr:
            #  Hypocenter location
            RR.setHypocenter(Sty, SuD)

            #  Rupture velocity pattern and define Onset times
            locVr = [LcllVr[sii], LcllVr[sii], 1 + Hvr]
            sucV = RR.setRuptVeloc(LCVrrI[sii], stdVr, limsVr, locVr, rnuc, Vor, gbou, Vfr, cVpVr=cVpVr)
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
            succ=computeSource(sii, False)
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
    nreal = 100  # Number of realizations
    foldS = 'Sources/RS65B/'  # Output folder
    dll, dww = 2.75 * 100.0, 2.75 * 100.0  #  Grid spacing in the fault
    dtt = 1E-3  # Step time
    Mw = 6.5  # (np.log10(Mo) - 9.05) / 1.5
    Sty = 'SD'  # 'SD'#'SS' #--> SS: Strike-Slip SD:Slip-Dip
    SuD = 'SU'  # 'SS'#'CR' # SU #--> SS:'Strike-Slip' CR: Crustal-Dip SU: Subduction-Dip-Slip
    sufi = 'Si_' #  Suffix names of the sources
    regTr='Som'  # Regresion for average rise time
    # ------------------------------------------- Geometry inputs of the fault
    PoiR_I = np.asarray([0.5, 1.0])  # Relative location of a fix point (reference the left down corner)
    PoiL = np.asarray([0, 0, -3.2E3])  # Absolute location of PoiR
    tapPL_G = np.asarray([0.15, 0.15, 0.15, 0.15])  # Tapper definition [-x,+x,-y,+y]
    stri = 0  # Strike

    CVrr_I = 0.75  # Average rupture velocity coefficient with respect Vs
    stdVr = 0.25  # Std rupture velocity coefficient with respect Vs
    stAVr = 0.1  # std of Average rupture velocity coefficient with respect Vs
    Vor = 0.7  # Coeficient of reduction for Vr around the hypocenter
    rnuc = 1500  # nucleation radious
    Vfr = 0.4  # Coeficient of reduction for Vr close to the edges
    gbou = 1000  # Gap to the borders where Vr decays

    randHyp = True  #  Random selection of the hypocenter? if not, PoiR defines the hypocenter
    randUmax = True  #  Random selection of the maximal slip
    randOri = True  # Randomlly set the orientation based on the fault type
    randLW = (True, True)  # (strike dir, dip dir) [degrees]: Randomlly set the fault dimension based on the fault type
    addSSV = True  # Add small scale variations
    #----------------------------------------------  Fixed inputs
    H = 0.77  # Mai and Beroza, 2002
    Hvr = -0.3
    stRa = np.deg2rad(15)  # --radians-- standard deviation of rake variations (Graves Pitarka, 2010)
    N = 4  # Sharpness transition parameter
    cUTr = [0.50, 0.90]  # Correlation Correlation slip-tr [lim_inf, lim_sup]
    cVpVr = [0.25, 0.65]  # Correlation vpeak, rupture veloity
    KtacM = 0.05  # Tpeak average
    vpkMax = 6.5  # Maximal slip rate
    limsVr = [0.002, 1]  # Limits of ratio Vr/Vs in th fault plane
    #  ----------------------------------------------------------------------------------------------
    #  ----------------------------------------------------------------------------------------------  End inputs
    nmit=25# max number of iterations to solve a source
    if not path.exists('%s' % foldS): makedirs(foldS[:-1])
    ncore = cpu_count()

    Mo = 10 ** (Mw * 1.5 + 9.05)
    LI, WI, Lstri, Ldip, Lrake = rg.computeGeometryParams(nreal, Mw,Sty, SuD, stri, randLW, randOri)
    LI = np.round(1E3 * LI / dll) * dll
    WI = np.round(1E3 * WI / dww) * dww

    if randUmax: LVumaxI = 10**(0.1 * np.random.randn(nreal))

    Lcll, Lcww = rg.computeVKParams(LI, WI, Mw, Sty)
    LcllVr = Lcll.copy()
    LcwwVr = Lcww.copy()

    trp_I, strp = rg.computeAveRiseTime(regTr, Mw)
    LtrpI = trp_I * 10 ** (np.random.randn(nreal) * strp)

    LCVrrI = CVrr_I + np.random.randn(nreal) * stAVr
    LCVrrI[LCVrrI < 0.6] = 0.6
    LCVrrI[LCVrrI > 0.9] = 0.9

    #list(map(computeSource, [0]))

    func = partial(computeSource, patt=True)
    tic = time()
    with Pool(ncore) as pool:
        list(pool.map(func, np.arange(0, nreal)))#pool.#range(niter)

        pool.close()
    print('%f [s]' %(time()-tic))
    print('Fin :)')
    print('Fin!!!')
