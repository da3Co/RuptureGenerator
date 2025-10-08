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
def computeSource(sii, Inp, patt=True):
    '''

    :param sii: Index of the simulation
    :param Inp: Inptus lists
    :param patt: is the main branch
    :return: Creates a rupture process and write it to hdf5 files
    '''

    (LMo, LI, WI, Lstri, Ldip, Lrake, Lcll, Lcww, LtrpI, LCVrrI, LcllVr, LcwwVr, dll, dww, PoiR_I, PoiL, tapPL_G,
        Vor, rnuc, Vfr, gbou, stRa, H, Hvr, cUTr, cVpVr, KtacM, vpkMax, limsVr, nmit, randUmax, addSSV, Sty,
         SuD, LMec, stdVr, dtt, foldS, sufi, randHyp, LSty, LSuD, LMec) = Inp

    np.random.seed(np.random.randint(1, 10E3) * (1 + sii))
    RR = rg.Rupture(LMo[sii], LI[sii], WI[sii], LSty[sii], LSuD[sii], LMec[sii], sii)

    RR.assingLocationsOrientations(Lstri[sii], Ldip[sii], Lrake[sii], dll, dww, PoiR_I, LPoiL[sii])
    sucS, sucTr, sucV = False, False, False

    _, Vs, Rho = CSuelo(RR.Pos[2])
    RR.assingMaterialProp(Vs, Rho)

    loc = [Lcll[sii], Lcww[sii], 1 + H]
    #  Slip pattern definition
    sucS = RR.setSlipPattern(loc, tapPL_G, None)#if randUmax:
    #else: RR.setSlipPattern(loc, tapPL_G,umax )

    if sucS:
        #  Rakes variation definition
        RR.setRakesVariations(stRa)

        sea = True
        isea = -1
        while sea and isea < 4:
            #  Rise Time definition
            sucTr = RR.setRiseTimePattern(loc, KtacM, LtrpI[sii], cUTr, vpkMax, cmax=500)

            if sucTr:
                #  Hypocenter location
                if randHyp: RR.setHypocenter()
                else: RR.setHypocenter((int(np.round(PoiR_I[0]*RR.nll.size)), int(np.round(PoiR_I[1]*RR.nww.size))))

                #  Rupture velocity pattern and define Onset times
                locVr = [LcllVr[sii], LcllVr[sii], 1 + Hvr]
                sucV = RR.setRuptVeloc(LCVrrI[sii], stdVr, locVr, rnuc, Vor, gbou, Vfr,
                                       limsVr=limsVr, cVpVr=cVpVr, cmax=1000)

                if sucV:
                    sea=False
                    #  Compute STF at each subfault
                    RR.generateSTFOpt(dtt, addSSV)

                    #  Write sources
                    RR.writePickle(foldS + '%s%d.pickle' % (sufi, sii))
            isea += 1

    csus=sucS and sucTr and sucV
    if not (csus) and patt:
        ii = 0
        succ = False
        while ii < nmit and not (succ):
            ii += 1
            succ = computeSource(sii, Inp, False)
            # print('Fail try: [%d, %r, %r, %r, %r]' % (sii, sucS, sucTr, sucV, hasattr(RR, 'To')))
        if not (succ):
            print('Rupture no generated: [%d, Slp:%r, Tr:%r, Vr:%r, %r]' % (sii, sucS, sucTr, sucV, hasattr(RR, 'To')))

    return csus

def CSuelo(zz):
    '''

    :param zz: matrix of depth
    :return: Must return the Vp(no need), Vs, Rho at each point. In this case a 1D layerd geologic model defined in fgeo
    '''
    fgeo = 'geo/1DLA.txt'  # File with mechanical soil properties description
    Da = np.loadtxt(fgeo, skiprows=1, delimiter=',')
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
    #  Inputs ----------------------------------------------------------------------------------------------------------
    nreal = 22  # Number of realizations
    foldS = 'Results/Example2/Sources/'  # Output folder
    dll, dww = 400.0, 400.0  #  Grid spacing in the fault
    dtt = 1E-3  # Step time
    LMw = np.ones(nreal) * 6.5  # (np.log10(Mo) - 9.05) / 1.5
    Sty = 'DS'  # 'DS'#'SS' #--> SS: Strike-Slip DS:Dip-Slip
    SuD = 'CR'  # 'SS'#'CR' # SU #--> SS:'Strike-Slip' CR: Crustal-Dip SU: Subduction-Dip-Slip
    Mec = 'NS'  # Mechanism to find dip and rake randmolly. SS: Strike-Slip NS: Normal fault RS: Reverse fault
    sufi = 'Si_' #  Suffix names of the sources
    regTr='M&H' # Regression to compute the average rise time; 'Som': Somerville et al. (1999),
    #               'Miy': Miyake et al, 2003 'M&H':  # Melgar and Hayes, 2003, 'Gus': Gusev and Chebrov, 2019
    # ------------------------------------------- Geometry inputs of the fault
    PoiR_I = np.asarray([0.5, 1.0])  # Relative location of a fix point (reference the left down corner)
    PoiL = np.asarray([0, 0, -5.2E3])  # Absolute location of PoiR
    tapPL_G = np.asarray([0.15, 0.15, 0.15, 0.15])  # Tapper definition [-x,+x,-y,+y]
    strike = 0  # Strike

    CVrr_I = 0.75  # Average rupture velocity coefficient with respect Vs
    stdVr = 0.25  # Std rupture velocity coefficient with respect Vs
    stAVr = 0.1  # std of Average rupture velocity coefficient with respect Vs
    Vor = 0.7  # Coeficient of reduction for Vr around the hypocenter
    rnuc = 1500  # nucleation radious
    Vfr = 0.4  # Coeficient of reduction for Vr close to the edges
    gbou = 1000  # Gap to the borders where Vr decays

    randHyp = True  #  Random selection of the hypocenter? if not, PoiR defines the hypocenter
    randUmax = True  #  Random selection of the maximal slip
    randLW = (True, True)  # (strike dir, dip dir) [degrees]: Randomlly set the fault dimension based on the fault type
    addSSV = True  # Add small scale variations
    #----------------------------------------------  Fixed inputs
    H = 0.77  # Mai and Beroza, 2002
    Hvr = -0.3
    stRa = np.deg2rad(15)  # --radians-- standard deviation of rake variations (Graves Pitarka, 2010)
    cUTr = [0.50, 0.90]  # Correlation crrelation slip-tr [lim_inf, lim_sup]
    cVpVr = [0.25, 0.65]  # Correlation vpeak, rupture veloity
    KtacM = 0.05  # Tpeak average
    vpkMax = 6.5  # Maximal slip rate
    limsVr = [0.002, 1]  # Limits of ratio Vr/Vs in th fault plane
    #  ----------------------------------------------------------------------------------------------
    #  ----------------------------------------------------------------------------------------------  End inputs
    nmit=25# max number of iterations to solve a source
    if not path.exists('%s' % foldS): makedirs(foldS[:-1])
    ncore = cpu_count()

    Ori=np.asarray(list(map(lambda ii: rg.computeOrientationParams(Mec), range(nreal))))
    Lstri, Ldip, Lrake = strike*np.ones(nreal), Ori[:,0], Ori[:,1]

    LDi = np.asarray(list(map(lambda ii: rg.computeGeometryParams(LMw[ii], Sty, SuD, randLW), range(nreal))))
    LI = np.round(LDi[:, 0] / dll) * dll
    WI = np.round(LDi[:, 1] / dww) * dww

    Lcll, Lcww = rg.computeVKParams(LI, WI, LMw, Sty)
    LcllVr = Lcll.copy()
    LcwwVr = Lcww.copy()

    trp_I, strp = rg.computeAveRiseTime(regTr, LMw)
    LtrpI = trp_I * 10 ** (np.random.randn(nreal) * strp)

    LCVrrI = CVrr_I + np.random.randn(nreal) * stAVr
    LCVrrI[LCVrrI < 0.6] = 0.6
    LCVrrI[LCVrrI > 0.9] = 0.9

    LMo = 10 ** (LMw * 1.5 + 9.05)
    LSty = [Sty] * nreal
    LSuD = [SuD] * nreal
    LMec = [Mec] * nreal
    LPoiL = np.repeat([PoiL], nreal, axis=0)

    Inp=(LMo, LI, WI, Lstri, Ldip, Lrake, Lcll, Lcww, LtrpI, LCVrrI, LcllVr, LcwwVr, dll, dww, PoiR_I, LPoiL, tapPL_G,
        Vor, rnuc, Vfr, gbou, stRa, H, Hvr, cUTr, cVpVr, KtacM, vpkMax, limsVr, nmit, randUmax, addSSV, Sty,
         SuD, LMec, stdVr, dtt, foldS, sufi, randHyp, LSty, LSuD, LMec)
    func = partial(computeSource, Inp=Inp, patt=True)

    #list(map(func, [0]))
    tic = time()
    with Pool(ncore) as pool:
        list(pool.map(func, np.arange(0, nreal)))#pool.#range(niter)

        pool.close()
    print('%f [s]' %(time()-tic))
    print('Fin :)')
    print('Fin!!!')
