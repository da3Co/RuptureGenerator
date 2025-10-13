# -*- coding: utf-8 -*-
"""
@author: David Alejandro Castro Cruz
da.castro790@uniandes.edu.co
cite:
Insitution: King Abdullah University of Science and Technology

Please cite us:
David Castro-Cruz, Paul Martin Mai, A new kinematic rupture generation technique and its application,
Geophysical Journal International, Volume 243, Issue 3, December 2025, ggaf385, https://doi.org/10.1093/gji/ggaf385
"""

import numpy as np
from scipy.stats import gamma, weibull_min, norm, truncexpon
from scipy.optimize import fsolve, minimize
from fteikpy import Eikonal2D
from functools import partial
import pickle
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#mpl.use('TkAgg')  # interactive mode works with this, pick one

def addHFSlip(Mo, Dks, Dkd, Vs, Rho, dll, dww, loc, ade, D_u, cdd, ws, phaseI, F, lTapSlp=0.8, cmax=4000):
    val = True
    ccS = -1

    # slip computation
    while val:
        theta0 = np.random.rand(Dks.shape[0], Dkd.shape[1]) * 2 * np.pi
        theta0[0, 0] = 0.0
        phaseS = np.exp(theta0 * 1j) * (1 - F) + phaseI * F

        Slp = VKfield2D([loc[0], loc[1], loc[2]], Dks, Dkd, phaseS)

        #  Transformation
        Slp[np.unravel_index(np.argsort(Slp.flatten()), Slp.shape)] = np.sort(D_u.rvs(size=Slp.size))

        #  Tapper check
        Swt = Slp.copy()
        Slp = Tapper(Swt, cdd, ws, ade)

        ccS += 1
        val = not np.sum(Slp) / np.sum(Swt) > lTapSlp and ccS < cmax

    return Slp * Mo / np.sum(dll * dww * Slp * Rho * Vs ** 2), np.logical_not(val), theta0

def computeGeometryParams(Mw, Sty, SuD, randLW, fa=1.15, th=2.5, mar=1.1):
    '''
    Thingbaijam et al., 2017
    :param Mw: -float-
    :param Sty: -string- Style of the source SS: Strike-Slip DS:Dip-Slip
    :param SuD: -string- Type of slip-dip fault SS:'Strike-Slip' CR: Crustal-Dip SU: Subduction-Dip-Slip
    :param randLW: -boolean- Allow L and W randomness
    :param fa: complete length per effective length ratio
    :param th: threshold in std units to no consider the realization and generating a new
    :param mar: Maximal aspect ratio W/L
    :return: length and with in meters
    '''
    #  Dimension Randomness
    if Sty == 'SS':
        L_I = fa * 10 ** (-2.943 + 0.681 * Mw)
        W_I = fa * 10 ** (-0.543 + 0.261 * Mw)
        if randLW[0]:
            LI = L_I * 10 ** (np.random.randn() * 0.151)
            if LI < L_I * 10 ** (- th * 0.151) or LI > L_I * 10 ** (th * 0.151): LI = L_I

        else:
            LI = L_I
        if randLW[1]:
            WI = W_I * 10 ** (np.random.randn() * 0.105)
            if WI < W_I * 10 ** (- th * 0.105) or WI > W_I * 10 ** (th * 0.105): WI = W_I

        else:
            WI = W_I

    elif Sty == 'DS':
        if SuD == 'CR':
            L_I = fa * 10 ** (-2.693 + 0.614 * Mw)
            W_I = fa * 10 ** (-1.669 + 0.435 * Mw)
            if randLW[0]:
                LI = L_I * 10 ** (np.random.randn() * 0.083)
                if LI < L_I * 10 ** (- th * 0.083) or LI > L_I * 10 ** (th * 0.083): LI = L_I

            else:
                LI = L_I
            if randLW[1]:
                WI = W_I * 10 ** (np.random.randn() * 0.087)
                if WI < W_I * 10 ** (- th * 0.087) or WI > W_I * 10 ** (th * 0.087): WI = W_I
            else:
                WI = W_I

        elif SuD == 'SU':

            L_I = fa * 10 ** (-0.880 + 0.366 * Mw)  # 10 ** (-2.412 + 0.583 * Mw)
            W_I = fa * 10 ** (-2.412 + 0.583 * Mw)  # 10 ** (-0.880 + 0.366 * Mw)
            if randLW[0]:
                LI = L_I * 10 ** (np.random.randn() * 0.107)
                if LI < L_I * 10 ** (- th * 0.107) or LI > L_I * 10 ** (th * 0.107): LI = L_I

            else:
                LI = L_I

            if randLW[1]:
                WI = W_I * 10 ** (np.random.randn() * 0.099)
                if WI < W_I * 10 ** (- th * 0.099) or WI > W_I * 10 ** (th * 0.099): WI = W_I

            else:
                WI = W_I

        else:  # Normal events
            L_I = fa * 10 ** (-1.722 + 0.485 * Mw)
            W_I = fa * 10 ** (-0.829 + 0.323 * Mw)
            if randLW[0]:
                LI = L_I * 10 ** (np.random.randn() * 0.128)
                if LI < L_I * 10 ** (- th * 0.128) or LI > L_I * 10 ** (th * 0.128): LI = L_I

            else:
                LI = L_I
            if randLW[1]:
                WI = W_I * 10 ** (np.random.randn() * 0.128)
                if WI < W_I * 10 ** (- th * 0.128) or WI > W_I * 10 ** (th * 0.128): WI = W_I
            else:
                WI = W_I

    else:
        print('Error no defined scale relation')

    if WI > mar * LI: WI = mar * LI

    return 1E3*LI, 1E3*WI

def computeUmax(uni, randUmax=True):
    '''

    :param uni: -array- average slip
    :param randUmax: -boolean- add randomness to umax
    :return:
    '''

    if randUmax:
        ra=1
        while ra>0.4:
            umax = 10 ** (0.95 * np.log10(uni) + 0.62 + np.random.randn(uni.size))  # maximal slip
            ra=uni/umax
    else:
        umax = 10 ** (0.95 * np.log10(uni) + 0.62)
    return umax
def computeOrientationParams(Mec):
    '''

    :param Mec: Mechanism SS: Strike-Slip NS: Normal fault RS: Reverse fault
    :return: dip, rake (degrees)
    '''
    if Mec == 'SS':
        dip = 90 - 15 * np.fabs(np.random.randn())
        while dip<40:
            dip = 90 - 15 * np.fabs(np.random.randn())
        rake = 60 * np.random.rand() - 30 - 180*np.random.binomial(1, 0.5)
    elif Mec == 'NS':
        dip = 55 * np.random.rand() + 30
        rake = 15 * np.random.randn()-90
        while rake < -150 or rake > -30:
            rake = 15 * np.random.randn()-90

    elif Mec == 'RS':
        dip = 70 * np.random.rand() + 10
        rake = 15 * np.random.randn() + 90
        while rake > 150 or rake < 30:
            rake = 15 * np.random.randn() + 90
    else:
        dip, rake = np.nan, np.nan
        print('Error no defined scale relation to determine dip and rake')

    return dip, rake

def computeSmallScalUni(timi):
    '''
    Compute the small scale variation between Yoffe function and a realistic STF, one vector time
    :param timi: time definition, linear equal spaced
    :return:
    '''
    ktt = np.fft.rfftfreq(timi.size, timi[1]-timi[0])

    theta0 = np.random.rand(ktt.size) * 2 * np.pi
    phaseT = np.exp(theta0 * 1j)
    return RF_SCV_1dVK(ktt, 0.089, phaseT)
def computeSlip(Mo, Dks, Dkd, Vs, Rho, dll, dww, loc, ade, D_u, cdd, ws, lTapSlp=0.8, cmax=4000):
    '''

    :param Mo: Seismic moment
    :param Dks: strike direction wave number matrix
    :param Dkd: dip direction wave number matrix
    :param Vs: Shear velocity matrix
    :param Rho: Density matrix
    :param dll: strike direction step grid
    :param dww: dip direction step grid
    :param loc: [cll, cww, 1+H]
    :param ade:
    :param D_u: Slip distribution object scipy
    :param cdd:
    :param ws:
    :param lTapSlp: minimial similitud after tappering to consider good the realization
    :param cmax: max number of tries
    :return:
    '''
    val = True
    ccS = -1
    Slp=np.empty(Vs.shape)
    # slip computation
    while val:
        theta0 = np.random.rand(Dks.shape[0], Dkd.shape[1]) * 2 * np.pi
        theta0[0, 0] = 0.0
        EphaseS = np.exp(theta0 * 1j)

        Fld = VKfield2D([loc[0], loc[1], loc[2]], Dks, Dkd, EphaseS)

        #  Transformation
        Slp[np.unravel_index(np.argsort(Fld.flatten()), Slp.shape)] = np.sort(D_u.rvs(size=Slp.size))

        #  Tapper check
        Swt = Slp.copy()
        Slp = Tapper(Swt, cdd, ws, ade)

        ccS += 1
        val = not np.sum(Slp) / np.sum(Swt) > lTapSlp and ccS < cmax

        if val:
            #  Transformation
            Slp[np.unravel_index(np.argsort(-Fld.flatten()), Slp.shape)] = np.sort(D_u.rvs(size=Slp.size))

            #  Tapper check
            Swt = Slp.copy()
            Slp = Tapper(Swt, cdd, ws, ade)

            val = not np.sum(Slp) / np.sum(Swt) > lTapSlp and ccS < cmax
    return Slp * Mo / np.sum(dll * dww * Slp * Rho * Vs ** 2), np.logical_not(val), theta0

def computeSlipFromTheta(theta0, Mo, Dks, Dkd, Vs, Rho, dll, dww, loc, ade, D_u, cdd, ws, lTapSlp=0.8):

    # slip computation
    EphaseS = np.exp(theta0 * 1j)

    Slp = VKfield2D([loc[0], loc[1], loc[2]], Dks, Dkd, EphaseS)

    #  Transformation
    Slp[np.unravel_index(np.argsort(Slp.flatten()), Slp.shape)] = np.sort(D_u.rvs(size=Slp.size))

    #  Tapper check
    Swt = Slp.copy()
    Slp = Tapper(Swt, cdd, ws, ade)


    return Slp * Mo / np.sum(dll * dww * Slp * Rho * Vs ** 2)

def computeVKParams(LI, WI, LMw, Sty, th=2.5):
    '''

    :param LI: -array- Length of each realization
    :param WI: -array- Width of each realization
    :param LMw: -array- Magnitude earthquake
    :param Sty: -string- Style of the source SS: Strike-Slip DS:Dip-Slip
    :param th: -float- threshold to keep a realization in terms of std, outside of it the value is remplaced
    :return:
    '''
    niter = LI.size
    var=np.random.randn(niter)
    res=np.any([var<-th, var>th], axis=0)
    while np.any(res):
        var[res] = np.random.randn(np.sum(res))
        res = np.any([var < -2.5, var > 2.5], axis=0)
    if Sty == 'SS':
        Lcll = 2 * np.pi * np.mean([10**(-2.928 + 0.588 * LMw), 1.855 + 0.341*LI*1E-3,
                                -4.870 + 0.741*WI*1E-3], axis=0) * 1E3
        Lcll *= 10 ** (var * 0.19)
    elif Sty =='DS':
        Lcll = 2 * np.pi * np.mean([10 ** (-2.433 + 0.492 * LMw), 1.096 + 0.314 * LI * 1E-3,
                                    1.832 + 0.449 * WI * 1E-3], axis=0) * 1E3
        Lcll *= 10 ** (var * 0.14)

    else:
        Lcll = 2 * np.pi * np.mean([10 ** (-2.595 + 0.527 * LMw), 1.541 + 0.326 * LI * 1E-3,
                                    6.072 + 0.416 * WI * 1E-3], axis=0) * 1E3
        Lcll *= 10 ** (var * 0.19)

    Lcww = Lcll.copy()

    return Lcll, Lcww

def computeAveRiseTime(reg, Mw):
    '''

    :param reg: Regression to compute the average rise time; 'Som': Somerville et al. (1999), 'Miy': Miyake et al, 2003
                'M&H':  # Melgar and Hayes, 2003, 'Gus': Gusev and Chebrov, 2019
    :param Mw:
    :return: expected average rise time, std of the regression
    '''
    if reg == 'Som': # Somerville et al. (1999)
        Mo = 10 ** (Mw * 1.5 + 9.05)
        trp = 2.03E-9 * ((Mo * 1E7) ** (1 / 3))  # Average riste time10**(-3.01+0.498*Mw)#
        strp = 0.2
    elif reg == 'Miy':  # Miyake et al, 2003
        trp = 10 ** (-3.34 + 0.5 * Mw)  # 10**(-2.66+0.439*Mw)
        strp = 0.1
    elif reg == 'M&H':  # Melgar and Hayes, 2017
        trp = 10 ** (-2.66 + 0.439 * Mw)  # 10**(-2.66+0.439*Mw)
        strp = 0.15
    elif reg == 'Gus':  # Gusev and Chebrov, 2019
        trp = 10 ** (-3.01 + 0.498 * Mw)  # 10**(-2.66+0.439*Mw)
        strp = 0.3
    else:
        print('No-scale for average rise time')
        print(reg)
        trp = np.nan
        strp = np.nan

    return trp, strp

def costWVr(par, limI, CVrr, stdVr):
    mm, va = weibull_min.stats(par[0], loc=limI, scale=par[1])
    return (1 - CVrr - mm) ** 2 + (stdVr ** 2 - va) ** 2

def estimateVpeak(Slip, Trr, tac):
    return 0.94818906 * Slip / (tac ** 0.55133027 * Trr ** 0.45673853)

def RF_SCV_1dVK(ktt, std, phase):
    '''

    :param ktt: frequency
    :param std: standard deviation final SCV
    :param phase: phase to create SCV (e^...)
    :return:
    '''
    VK = 1 / (1 + (ktt * 0.53805486) ** 2) ** (0.5 * (0.76413772 + 0.5))
    nn = np.fft.irfft(VK * phase)
    nn = nn - np.mean(nn)
    nn *= std / np.std(nn)
    return nn

def Tapper(SlpO, cdd, ws, ade):
    Slp=SlpO.copy()

    Slp[:cdd[0]] *= np.transpose(np.tile(ws[0][:cdd[0]], (Slp.shape[1], 1)))
    if ade[0] == 0 and cdd[1] > 0:
        Slp[-cdd[1]:] *= np.transpose(np.tile(ws[1][cdd[1]:], (Slp.shape[1], 1)))
    else:
        Slp[-(cdd[1] + ade[0]):-ade[0]] *= np.transpose(np.tile(ws[1][cdd[1]:], (Slp.shape[1], 1)))
        Slp[-ade[0]:] = 0.0

    Slp[:, :cdd[2]] *= np.tile(ws[2][:cdd[2]], (Slp.shape[0], 1))
    if ade[1] == 0 and cdd[3] != 0:
        Slp[:, -cdd[3]:] *= np.tile(ws[3][cdd[3]:], (Slp.shape[0], 1))
    elif cdd[3] != 0:
        Slp[:, -(cdd[3] + ade[1]):-ade[1]] *= np.tile(ws[3][cdd[3]:], (Slp.shape[0], 1))
        Slp[:, -ade[1]:] = 0.0
    return Slp

def VKfield2D(loc, Dks, Dkd, Ephase):
    '''

    :param loc: [length correlation L, length correlation W, H+1]
    :param Dks: Wave numbers in strike direction
    :param Dkd: Wave numbers in dip direction
    :param ExpPhase: exp(i*theta)
    :return: computed Slip
    '''
    sK = (loc[0] * Dks) ** 2 + (loc[1] * Dkd) ** 2
    Sg = np.sqrt(1 / (1 + sK) ** loc[2])

    return np.fft.irfft2(Sg * Ephase)

def Yoffe_traingleReg(timiR, to, Tr, tPic, tol=0.0):
    '''
    returns the slip_rate and value for given set of input for traingle based regulairzed Yoffe function
    :param time_onslip: time in secs
    :param to: time of triggering
    :param Tr: Rise time in secs
    :param tPic: peak time (see Tinti et al. 2005 as T_{acc}). Ts=min(time_acc,Tdu/3)
    :param tol: minimum assigned slip
    :return:

    % See  Tinti et al., 2005
    % Modified to python David Castro Cruz 2022
    '''
    timi = timiR-to
    Ts = tPic/1.27
    Tdu = Tr + 2.0 * Ts
    Ts = min([Ts, Tdu/3])

    kappa = 2.0 / (np.pi * Tr * Ts ** 2)
    yoffe = np.zeros(timi.size)
    C=[np.empty(timi.size) for ii in range(6)]


    C[0] = (0.5 * timi + 0.25 * Tr) * np.sqrt(timi * (Tr - timi)) + (timi * Tr - Tr ** 2) * \
         np.arcsin(np.sqrt(timi / Tr)) - 0.75 * Tr ** 2 * np.arctan(np.sqrt((Tr - timi) / timi))

    C[1] = 0.375 * np.pi * Tr ** 2

    C[2] = (Ts - timi - 0.5 * Tr) * np.sqrt((timi - Ts) * (Tr - timi + Ts)) + Tr * 2*(Tr - timi + Ts) * \
           np.arcsin(np.sqrt((timi - Ts) / Tr)) + 1.5 * (Tr ** 2) * np.arctan(np.sqrt((Tr - timi + Ts) / (timi - Ts)))

    C[3] = (-Ts + 0.5 * timi + 0.25 * Tr) * np.sqrt((timi - 2 * Ts) * (Tr - timi + 2 * Ts)) + \
        Tr * (-Tr + timi - 2 * Ts) * np.arcsin(np.sqrt((timi - 2 * Ts) / Tr)) - 0.75 * Tr ** 2 * np.arctan(
            np.sqrt((Tr - timi + 2 * Ts) / (timi - 2 * Ts)))

    C[4] = 0.5 * np.pi * Tr * (timi - Tr)

    C[5] = 0.5 * np.pi * Tr * (2*Ts - timi + Tr)


    ik = np.where(np.all([timi >= 0, timi < Ts], axis=0))[0]
    yoffe[ik] = C[0][ik] + C[1]

    if Tr > 2 * Ts:
        ik = np.where(np.all([timi >= Ts, timi < 2*Ts], axis=0))[0]
        yoffe[ik] = C[0][ik] - C[1] + C[2][ik]

        ik = np.where(np.all([timi >= 2 * Ts, timi < Tr], axis=0))[0]
        yoffe[ik] = C[0][ik] + C[2][ik] + C[3][ik]

        ik = np.where(np.all([timi >= Tr, timi < Tr + Ts], axis=0))[0]
        yoffe[ik] = C[4][ik] + C[2][ik] + C[3][ik]

    else:
        ik = np.where(np.all([timi >= Ts, timi < Tr], axis=0))[0]
        yoffe[ik] = C[0][ik] - C[1] + C[2][ik]

        ik = np.where(np.all([timi >= Tr, timi < 2*Ts], axis=0))[0]
        yoffe[ik] = C[4][ik] + C[2][ik] - C[1]

        ik = np.where(np.all([timi >= 2 * Ts, timi < Tr + Ts], axis=0))[0]
        yoffe[ik] = C[4][ik] + C[2][ik] + C[3][ik]

    ik = np.where(np.all([timi >= Tr + Ts, timi < Tr + 2 * Ts], axis=0))[0]
    yoffe[ik] = C[3][ik] + C[5][ik]

    yoffe[yoffe<tol] = tol
    return kappa*yoffe
class Rupture(object):

    def __init__(self, Mo, L, W, Sty, SuD=None, Mec=None, index=0):
        '''

        :param Mo: -float- Seismic moment
        :param L: -float- Length in the strike direction in meters
        :param W: -float- Width in the dip direction in meters
        :param index: -int- Source Index
        :param Sty: -String- Type of fault (--> SS: Strike-Slip, DS:Dip-Slip)
        :param SuD: -String- Type of fault2 (--> SS or None: Strike-Slip CR: Crustal-Dip SLip SU: Subduction-Dip)
        :param Mec: -String- Type of fault3 (--> SS or None: Strike-Slip NS: Normal Slip RS: Reverse)
        '''
        self.Mo = Mo
        self.L = L
        self.W = W
        self.Mw = (np.log10(Mo)-9.05)/1.5
        self.index = index
        if Sty=='SS': self.Sty = (Sty, Sty, Sty)
        elif Sty=='DS':
            self.Sty = (Sty, SuD, Mec)
        elif not Sty is None:
            print('The fault style is not defined')

    def computeSmallScaleVar(self, timi):
        '''
        Compute the small scale variation between Yoffe function and a realistic STF
        :param timi: time definition, linear equal spaced
        :return:
        '''
        ktt = np.fft.rfftfreq(timi.size, timi[1]-timi[0])
        kwt = np.fft.fftfreq(self.Slip.shape[1], self.dww)

        theta0 = np.random.rand(self.Dks.shape[0], kwt.size, ktt.size) * 2 * np.pi
        theta0[0, 0, 0] = 0.0
        phaseT = np.exp(theta0 * 1j)

        fu = partial(RF_SCV_1dVK, ktt, 0.089)  # trp Noisy ---------------  input noisy ----------
        return np.reshape(list(map(fu, phaseT.reshape(phaseT.shape[0] * phaseT.shape[1], phaseT.shape[2]))),
            (self.Slip.shape[0], self.Slip.shape[1], timi.size))

    def assingLocationsOrientations(self, strike, dip, rake, dll, dww, PoiR_I, PoiL):
        '''

        :param strike: -float- Strike in degrees
        :param dip: -float- Dip in degrees
        :param rake: -float- Average rake in degrees
        :param dll: space between subfault in strike direction
        :param dww: space between subfault in dip direction
        :param PoiR_I: [float, float] Relative location of a fix point (reference the left down corner)
        :param PoiL:  [float, float] # Absolute location of PoiR_I
        :return:
        '''

        self.dll = dll  # Spacing grid in strike direction
        self.dww = dww  # Spacing grid in dip direction

        PoiR = np.asarray([PoiR_I[0] * self.L, PoiR_I[1] * self.W])
        # Relative location of a fix point (reference the left down corner)

        #  Orientation definition
        an = np.radians(np.array([rake, strike, dip]))
        self.strike = strike  # Strike (Degrees)
        self.dip = dip  # Dip (Degrees)
        self.Arake = rake  # Rake (Degrees)

        #  Vectors Source

        #  Location future source
        self.vlon = np.asarray([np.sin(an[1]), np.cos(an[1]), 0])
        self.vwid = np.asarray([-np.cos(an[1]) * np.cos(an[2]), np.sin(an[1]) * np.cos(an[2]), np.sin(an[2])])
        self.vnor = np.cross(self.vlon, self.vwid)

        #  Define relative location
        self.nll, self.nww = np.arange(0, self.L, self.dll)+0.5*dll, np.arange(0, self.W, self.dww)+0.5*dww
        self.ade = [0, 0]
        if self.nww.size % 2 != 0:
            self.nww = np.append(self.nww[0] - self.dww, self.nww)
            self.ade[1] = 1
        self.Dll, self.Dww = np.meshgrid(self.nll, self.nww, indexing='ij')

        kw = np.fft.rfftfreq(self.Dll.shape[1], self.dww)
        kl = np.fft.fftfreq(self.Dll.shape[0], self.dll)
        self.Dks, self.Dkd = np.meshgrid(kl, kw, indexing='ij')

        #  Position
        Pos = np.empty((3,) + self.Dll.shape)
        for ndd in range(3):
            Pos[ndd] = (self.Dll - PoiR[0]) * self.vlon[ndd] + (self.Dww - PoiR[1]) * self.vwid[ndd] + PoiL[ndd]
        self.Pos = Pos

    def assingMaterialProp(self, Vs, Rho):
        '''

        :param Vs: -numpy array- Shear velocuty at each subfult
        :param Rho: -numpy array- Density at each subfult
        :return:
        '''
        self.Vs = Vs
        self.Rho = Rho

    def defineWeigthFunction(self, fLim, N=4):
        '''
        Defines the weight function between a predefined source and the new stochatic High frequency part
        :param fLim: maximal frequency definition of the initial source
        :param N: Sharpness transition parameter
        :return:
        '''

        VssP = np.mean(self.Vs)
        cl = VssP / fLim  # min([Vs / fmaxS, fe.attrs['LL'] / 2])
        cw = VssP / fLim  # min([Vs / fmaxS, fe.attrs['LW'] / 2])
        self.F = 1 / (1 + ((cl * self.Dks) ** 2 + (cw * self.Dkd) ** 2) ** N)

    def generateSTFOpt(self, dtt, SSVadd=True):
        '''
        Assing to STFs the final STF at each subfault, store in a optimize way (0, to,to+dt,..., to+1.2tr, tf)
        :param dtt: step of time
        :param SSVadd: -boolean- add Small scale Variations
        :return:
        '''

        tma = np.max(self.To + 2.0*self.Trise)
        shL = int(np.min(self.To)/dtt>1)
        timiT = np.arange(0, tma / dtt + 1) * dtt

        Mop = [[None]*self.Dll.shape[1] for ii in range(self.Dll.shape[0])]

        SM=np.zeros(timiT.size)
        Vmax=np.zeros(self.Dll.shape)
        for il in range(self.Dll.shape[0]):
            SMp = np.zeros(timiT.size)
            for iw in range(self.Dll.shape[1]):
                if self.Slip[il, iw] != 0:
                    tii = int(self.To[il, iw]/dtt)-shL
                    tff = tii + int((self.Trise[il, iw] + 3 * self.tac[il, iw]/1.27)/dtt)+1
                    stf = Yoffe_traingleReg(timiT[tii:tff] + dtt * 0.5, self.To[il, iw], self.Trise[il, iw],
                                            self.tac[il, iw])
                    vpp = np.max((self.Slip[il, iw] / (np.sum(stf) * dtt)) * stf)
                    while vpp > self.vpkMax:  # management of sub-faults with still too large vpeak
                        kk = np.random.rand() * 0.05 + 1
                        self.tac[il, iw] *= kk
                        self.Trise[il, iw] *= kk

                        tff = tii + int((self.Trise[il, iw] + 3 * self.tac[il, iw] / 1.27) / dtt) + 1
                        stf= Yoffe_traingleReg(timiT[tii:tff]+dtt*0.5, self.To[il,iw],self.Trise[il,iw],self.tac[il,iw])
                        vpp = np.max((self.Slip[il, iw] / (np.sum(stf) * dtt)) * stf)

                    while np.any(np.isnan(stf)):
                        kk = np.random.rand() * 0.05 + 1
                        self.Trise[il, iw] *= kk

                        tff = tii + int((self.Trise[il, iw] + 3 * self.tac[il, iw] / 1.27) / dtt) + 1
                        stf=Yoffe_traingleReg(timiT[tii:tff]+dtt*0.5, self.To[il,iw],self.Trise[il,iw], self.tac[il,iw])
                    re = np.where(stf > 0)[0]
                    if len(re) == 0:
                        stf[0] = 1

                    elif SSVadd:
                        re = np.append(re, np.arange(re[-1] + 1, re[-1]+ (0.05 * self.Trise[il, iw]) // dtt, dtype=int))
                        re = re[re < stf.size]
                        anw = np.ones(re.size)
                        nta = int(0.05 * self.Trise[il, iw] // dtt) + 1
                        if nta > 1: anw[-(nta - 1):] = 1 - np.arange(1, nta) / nta
                        SCV=computeSmallScalUni(timiT[:(re.size+1)])
                        stf[re] += SCV[:re.size]* np.max(stf[re]) * anw
                        stf[stf < 0] = 0.0
                    tff = tii + re[-1] + 1
                    moi = np.cumsum(stf[:(re[-1] + 1)]) * dtt
                    muA = self.dll * self.dww * self.Vs[il, iw] ** 2 * self.Rho[il, iw]

                    # muA = 1.0
                    Mop[il][iw] = [[tii, tff], self.Slip[il, iw] * muA * moi / moi[-1]]
                    SMp[tii:tff] += Mop[il][iw][1]
                    SMp[tff:] += Mop[il][iw][1][-1]

                    Vmax[il,iw]=np.max(np.gradient(Mop[il][iw][1],dtt))/muA
            SM += SMp
        self.SlipRateMax=Vmax
        self.LocalMo = Mop
        self.SeismicMoment = SM
        self.timiT=timiT

    def generateSTF(self, timi, SSV=None):
        '''
        Assing to STFs the final STF at each subfault
        :param SSV: array([ll, dd, tt]) matrix with the small scale variations
        :return:
        '''
        Mop = np.empty(list(self.Dll.shape) + [timi.size])
        dtt = timi[1] - timi[0]
        SSVadd = not (SSV is None)
        for il in range(self.Dll.shape[0]):
            for iw in range(self.Dll.shape[1]):
                if self.Slip[il, iw] != 0:
                    stf = Yoffe_traingleReg(timi + dtt * 0.5, self.To[il, iw], self.Trise[il, iw],self.tac[il, iw])
                    vpp = np.max((self.Slip[il, iw] / (np.sum(stf) * dtt)) * stf)
                    while vpp > self.vpkMax:  # management of sub-faults with still too large vpeak
                        kk = np.random.rand() * 0.05 + 1
                        self.tac[il, iw] *= kk
                        self.Trise[il, iw] *= kk

                        stf = Yoffe_traingleReg(timi+dtt*0.5, self.To[il,iw], self.Trise[il,iw], self.tac[il, iw])
                        vpp = np.max((self.Slip[il, iw] / (np.sum(stf) * dtt)) * stf)

                    while np.any(np.isnan(stf)):
                        kk = np.random.rand() * 0.05 + 1
                        self.Trise[il, iw] *= kk

                        stf = Yoffe_traingleReg(timi+dtt *0.5, self.To[il,iw], self.Trise[il,iw], self.tac[il,iw])
                    re = np.where(stf > 0)[0]
                    if len(re) == 0:
                        stf[0] = 1

                    elif SSVadd:
                        re = np.append(re, np.arange(re[-1] + 1, re[-1] + (0.05 * self.Trise[il, iw])//dtt, dtype=int))
                        re = re[re<stf.size]
                        anw = np.ones(re.size)
                        nta = int(0.05 * self.Trise[il, iw] // dtt) + 1
                        if nta > 1: anw[-(nta - 1):] = 1 - np.arange(1, nta) / nta
                        stf[re] += SSV[il, iw, re] * np.max(stf[re]) * anw
                        stf[stf < 0] = 0

                    moi = np.cumsum(stf) * dtt
                    muA = self.dll * self.dww * self.Vs[il, iw] ** 2 * self.Rho[il, iw]
                    # muA = 1.0
                    Mop[il, iw, :] = self.Slip[il, iw] * muA * moi / moi[-1]

                else:
                    Mop[il, iw, :] = np.zeros(timi.size)

        self.STFs = Mop * self.Mo / np.sum(Mop[:, :, -1])

    def setHypocenter(self, hypo=None):
        '''
        Computes the hypocenter following a probability distribution of Mai et al, 2005
        :param hypo: hypocenter, None means it computes randomlly
        :return:
        '''
        niteH=1
        if hypo is None:
            #  Hypocenter statistics
            if self.Sty[0] == 'SS':
                alphaM = 1.928
                betaM = 0.868

                alphaT = 0.446
                betaT = 1.551

                alphaZ = 0.626
                betaZ = 3.921
            elif self.Sty[0] == 'DS':
                alphaM = 2.216
                betaM = 0.623

                alphaT = 0.454
                betaT = 1.874

                if self.Sty[1] == 'CR':
                    alphaZ = 0.692
                    betaZ = 3.394
                elif self.Sty[1] == 'SU':
                    alphaZ = 12.658
                    betaZ = 0.034
                else:
                    alphaZ = 7.394
                    betaZ = 0.072

            else:
                alphaM = 2.210
                betaM = 0.748

                alphaT = 0.450
                betaT = 1.688

                alphaZ = 0.612
                betaZ = 3.353

            uva = self.Slip / np.mean(self.Slip)
            uma = self.Slip / np.max(self.Slip)
            #hxx = self.Dll/self.L - 0.5
            hxx = (self.Dll - 0.5 * self.tapPL[0]) / (self.L - 0.5 * self.tapPL[0] - 0.5 * self.tapPL[1])
            hzz = 1 - (self.Dww - 0.5 * self.tapPL[2]) / (self.W - 0.5 * self.tapPL[2] - 0.5 * self.tapPL[3])  #1-Dww/W

            dpp = np.ones(self.Slip.shape)
            dpp = dpp / np.sum(dpp)
            erF = np.inf

            dma= np.sqrt(np.max([self.Dll**2, (self.L - self.Dll)**2],axis=0)+np.max([self.Dww**2,(self.W-self.Dww)**2],
                                                                                     axis=0))
            ima = np.unravel_index(np.argmax(self.Slip), self.Dll.shape)
            rdmx = np.sqrt((self.Dll - self.Dll[ima]) ** 2 + (self.Dww - self.Dww[ima]) ** 2) / dma

            xx = np.linspace(0, 0.8, 1000)
            fts = np.sin(np.pi * xx / 0.8)
            fts = fts / (np.sum(fts) * xx[1])

            for inte in range(niteH):
                hist, bins = np.histogram(np.ravel(rdmx), int(self.Slip.size / 200),density=True, weights=np.ravel(dpp))
                hist[hist == 0] = 1 / self.Slip.size
                bb = 0.5 * (bins[1:] + bins[:-1])
                Tpdf = np.interp(rdmx, np.linspace(0, 0.8, 1000) + 0.04, fts, 0, 0)
                Rpdf = np.interp(rdmx, bb, hist)
                err = np.sum((Tpdf - Rpdf) ** 2)
                dpp *= Tpdf / Rpdf
                dpp = dpp / np.sum(dpp)

                hist, bins = np.histogram(np.ravel(uva), int(self.Slip.size / 200), density=True, weights=np.ravel(dpp))
                hist[hist == 0] = 1 / self.Slip.size
                bb = 0.5 * (bins[1:] + bins[:-1])
                Tpdf = gamma.pdf(uva, alphaM, loc=0, scale=betaM)
                Rpdf = np.interp(uva, bb, hist)
                err += np.sum((Tpdf - Rpdf) ** 2)
                dpp *= Tpdf / Rpdf
                dpp = dpp / np.sum(dpp)

                hist, bins = np.histogram(np.ravel(uma), int(self.Slip.size / 200), density=True, weights=np.ravel(dpp))
                hist[hist == 0] = 1 / self.Slip.size
                bb = 0.5 * (bins[1:] + bins[:-1])
                Tpdf = weibull_min.pdf(uma, betaT, loc=0, scale=alphaT)
                Rpdf = np.interp(uma, bb, hist)
                err += np.sum((Tpdf - Rpdf) ** 2)
                dpp *= Tpdf / Rpdf
                dpp = dpp / np.sum(dpp)

                hist, bins = np.histogram(np.ravel(hzz), np.unique(hzz),  # int(Slp.size / 200),
                                          density=True, weights=np.ravel(dpp))
                hist[hist == 0] = 1 / self.Slip.size
                bb = 0.5 * (bins[1:] + bins[:-1])

                if self.Sty[0] == 'DS' and self.Sty[1] != 'CR':
                    Tpdf = gamma.pdf(hzz, alphaZ, loc=0, scale=betaZ)
                else:
                    Tpdf = weibull_min.pdf(hzz, betaZ, loc=0, scale=alphaZ)

                Rpdf = np.interp(hzz, bb, hist)
                err += np.sum((Tpdf - Rpdf) ** 2)
                dpp *= Tpdf / Rpdf
                # dpp[np.any([hzz < 0, hzz > 1], axis=0)] = 0.0
                dpp = dpp / np.sum(dpp)

                hist, bins = np.histogram(np.ravel(hxx), np.unique(hxx),  # max([10, int(Slp.shape[0] / 10)]),
                                          density=True, weights=np.ravel(dpp))
                hist[hist == 0] = 1 / self.Slip.size
                bb = 0.5 * (bins[1:] + bins[:-1])
                Tpdf = norm.pdf(hxx, loc=0.5, scale=0.23)
                Rpdf = np.interp(hxx, bb, hist)
                err += np.sum((Tpdf - Rpdf) ** 2)
                dpp *= Tpdf / Rpdf
                # dpp[np.any([hxx<0, hxx>1], axis=0)]=0.0

                dpp = Tapper(Tapper(Tapper(Tapper(dpp, self.cdd, self.ws, self.ade), self.cdd, self.ws, self.ade),
                             self.cdd, self.ws, self.ade), self.cdd, self.ws, self.ade)
                dpp = dpp / np.sum(dpp)
                if err < erF:
                    dpF = dpp
                    erF = err

            dpF = dpF / np.sum(dpF)
            self.pHyp = dpF
            Hval = False
            while not (Hval):
                chyo = np.unravel_index(int(np.random.choice(self.Slip.size, 1, p=dpF.flatten())), self.Slip.shape)

                dma = np.sqrt(max([self.Dll[chyo] ** 2, (self.L - self.Dll[chyo]) ** 2]) + max([self.Dww[chyo] ** 2,
                                                                            (self.W - self.Dww[chyo]) ** 2]))
                Rhyp = np.sqrt((self.Dll - self.Dll[chyo]) ** 2 + (self.Dww - self.Dww[chyo]) ** 2) / dma

                ima = np.unravel_index(np.argmax(self.Slip), self.Slip.shape)
                drm = Rhyp[ima]

                Hval = drm < 0.8 and drm > 0.04 and np.min(Rhyp[self.Slip > 0.66 * self.Slip[ima]]) < 0.6
                Hval= Hval and np.min(Rhyp[np.all([self.Slip>0.33*self.Slip[ima],self.Slip<0.66*self.Slip[ima]],
                                                  axis=0)]) < 0.3

        else:chyo = hypo
        self.hypo = chyo

    def setOnsetTimes(self, sh=0.01):
        # Solve Eikonal at source
        eik = Eikonal2D(self.Vr, gridsize=(self.dll, self.dww))
        tt = eik.solve([self.Dll[self.hypo], self.Dww[self.hypo]])
        tooF = tt[:-1, :-1]
        self.To = tooF - np.min(tooF) + sh

    def setRakesVariations(self, stRa=0.2617993877991494, rand=True):
        an=np.radians(self.Arake)
        if rand:
            D_Ra = norm(loc=0, scale=stRa)

            #  Rake variations
            theta0 = np.random.rand(self.Dks.shape[0], self.Dkd.shape[1]) * 2 * np.pi
            theta0[0, 0] = 0.0
            EphaseRa = np.exp(theta0 * 1j)

            Ra = VKfield2D([self.cll, self.cww, self.H], self.Dks, self.Dkd, EphaseRa)
            Ra[np.unravel_index(np.argsort(Ra.flatten()), Ra.shape)] = np.sort(D_Ra.rvs(size=Ra.size))
            Ra += an
            self.rakes = Ra  # in radians
            self.phaseRakes = theta0
        else:
            self.rakes = an*np.ones(self.Dll.shape)  # in radians

    def setRiseTimePattern(self, loc, tacM, trp, cUTr=None, vpkMax=6.5, cmax=4000, lTapSlp=0.8, TrZo=None):
        '''

        :param loc: [float, float, float] [lengtCorrelation strike, lengtCorrelation dip, 1+Hurts exponent]
        :param tacM: -float- Tpeak average
        :param trp: -float- Average rise time
        :param cUTr: [float, float] correlation range between slip and rise time
        :param vpkMax: -float- maximal slip rate
        :param cmax: -int- maximal number of tries before report a fail
        :return:
        '''
        if TrZo is None: Bfa = np.ones(self.Dll.shape)
        else: Bfa = TrZo
        if cUTr is None: cUTr = [0.50, 0.90]

        self.trp = trp  # average rise time
        self.vpkMax = vpkMax  # Maximal slip rate
        ccTr = 0
        succ = False
        eta = np.linspace(0, 4, cmax + 1)
        F1 = VKfield2D(loc, self.Dks, self.Dkd, np.exp(self.phaseSlip * 1j))

        valS=True
        while valS and ccTr < cmax:
            theta0 = np.random.rand(self.Dks.shape[0], self.Dkd.shape[1]) * 2 * np.pi
            theta0[0, 0] = 0.0
            Ephase = np.exp(theta0 * 1j)
            Fld=VKfield2D(loc, self.Dks, self.Dkd, Ephase)

            cc=-1
            while valS and cc<3:
                ccTr += 0.25
                cc=cc+1
                Slp = eta[int(ccTr)] * F1 *(-1)**cc + Fld*(-1)**(cc//2)
                #  Transformation
                Slp[np.unravel_index(np.argsort(Slp.flatten()), Slp.shape)] = np.sort(self.D_u.rvs(size=Slp.size))

                #  Tapper check
                Swt = Slp.copy()
                Slp = Tapper(Swt, self.cdd, self.ws, self.ade)

                if not np.sum(Slp) / np.sum(Swt) > lTapSlp and ccTr < cmax:
                    Trr = np.sqrt(Slp)
                    #  correcting rise time distribution
                    Trr *= Bfa
                    Trr *= trp / np.mean(Trr[self.Slip > 0])
                    mcc = 0.02 * trp
                    Trr[Trr < mcc] = mcc

                    crt = np.corrcoef(self.Slip.flatten(), Trr.flatten())[0, 1]
                    eva0 = crt > cUTr[0] and crt < cUTr[1]

                    if eva0:
                        # coTr=True
                        # Vpeak
                        tac = Trr * (tacM / np.mean(Trr))  # Tpeak computation!!!
                        tac[tac < 0.01] = 0.01
                        maTr = Trr < 1.2 * tac
                        Trr[maTr] = 1.2 * tac[maTr]

                        self.tac = tac
                        self.vpk = estimateVpeak(self.Slip, Trr, tac)
                        succ = np.sum(self.vpk > vpkMax) / self.vpk.size < 0.02
                        if succ:
                            self.Trise = Trr
                            self.phaseTrise = theta0
                            valS = False

        return succ

    def setRiseTimePatternHF(self, loc, tacM, trp, phaseI, cUTr=None, vpkMax=6.5, cmax=4000):
        '''

        :param loc: [float, float, float] [lengtCorrelation strike, lengtCorrelation dip, 1+Hurts exponent]
        :param tacM: -float- Tpeak average
        :param trp: -float- Average rise time
        :param cUTr: [float, float] correlation range between slip and rise time
        :param vpkMax: -float- maximal slip rate
        :param cmax:
        :return:
        '''

        self.trp = trp  # average rise time
        self.vpkMax = vpkMax  # Maximal slip rate
        if cUTr is None: cUTr = [0.50, 0.90]
        ccTr = -1
        val=True
        while val:
            Trr, succ, phaseTr = addHFSlip(self.Mo, self.Dks, self.Dkd, self.Vs, self.Rho, self.dll, self.dww,
                                    loc, self.ade, self.D_u, self.cdd, self.ws, phaseI, self.F, 0.6)
            Trr = np.sqrt(Trr)

            #  correcting rise time distribution
            Trr *= trp / np.mean(Trr[self.Slip > 0])
            mcc = 0.02 * trp
            Trr[Trr < mcc] = mcc

            crt = np.corrcoef(self.Slip.flatten(), Trr.flatten())[0, 1]
            eva0 = crt > cUTr[0] and crt < cUTr[1]

            # coTr=True
            # Vpeak
            tac = Trr * (tacM / np.mean(Trr))  # Tpeak computation!!!
            tac[tac < 0.01] = 0.01
            maTr = Trr < 1.2 * tac
            Trr[maTr] = 1.2 * tac[maTr]

            self.tac = tac
            self.vpk = estimateVpeak(self.Slip, Trr, tac)
            eva1 = np.sum(self.vpk > vpkMax) / self.vpk.size < 0.02

            val = ccTr < cmax and (not (eva1) or not (eva0))  # if the errors afer few and loca, we manage
            # Those exceptions individualy
            ccTr += 1
        succ = np.logical_not(val)
        if succ:
            self.Trise = Trr
            self.phaseTrise = phaseTr
        return succ

    def setRuptVeloc(self, CVrr, stdVr, loc, rnuc=1500, Vor=0.7, gbou=1000, Vfr=0.4,
                     limsVr=None, maxT = 1E10, cVpVr = None, cmax=4000):
        '''

        :param CVrr: float average ratio Vr/Vs
        :param stdVr: std of the ratio Vr/Vs in the rupture
        :param loc: [float, float, float] [lengtCorrelation strike, lengtCorrelation dip, 1+Hurts exponent] for Vr/Vs
        :param rnuc: radious for the nucleation area
        :param Vor: Vr/Vs deduction at the hypocenter
        :param gbou:  Gap to the borders where Vr decays
        :param Vfr: Vr/Vs deduction at the borders
        :param limsVr: [float, float] maximal and minimal Vr/Vs ratio
        :param maxT: limit for maximal (onset time +1.2Tr)
        :param cVpVr: [float, float] Vr/Vs correlation range with vpeak
        :param cmax: maximal number of tries
        :return: if the function success
        '''

        if cVpVr is None: cVpVr = [0.2, 0.65]
        if limsVr is None: limsVr=[0.002, 1]
        #  Rupture propagation
        self.aVrr = CVrr  # Average ratio Vr/Vs
        self.locVr = loc
        ZoDvR = np.ones(self.Vs.shape)
        R = np.sqrt((self.Dll - self.Dll[self.hypo]) ** 2 + (self.Dww - self.Dww[self.hypo]) ** 2)
        res = R < rnuc
        ZoDvR[res] = Vor + (1.0 - Vor) * R[res] / rnuc

        ZoDv = np.ones(self.Vs.shape)
        res = self.Dll < gbou
        ZoDv[res] *= (1.0 - Vfr) * self.Dll[res] / gbou + Vfr
        res = self.Dww < gbou
        ZoDv[res] *= (1.0 - Vfr) * self.Dww[res] / gbou + Vfr
        res = self.nll[-1] - self.Dll < gbou
        ZoDv[res] *= (1.0 - Vfr) * (self.nll[-1] - self.Dll[res]) / gbou + Vfr
        res = self.nww[-1] - self.Dww < gbou
        ZoDv[res] *= (1.0 - Vfr) * (self.nww[-1] - self.Dww[res]) / gbou + Vfr

        #  Distribution 1-Vr/Vs
        coo = True
        ccVr = -2
        while coo and ccVr < cmax:
            resW = minimize(costWVr, np.asarray([1, 1]), args=(1-limsVr[1], CVrr, stdVr + 0.01 * (ccVr + 2)))
            coo = not (resW.success)
            ccVr += 1
        if not(coo):
            self.D_Vr = weibull_min(resW.x[0], loc=1-limsVr[1], scale=resW.x[1])

            succ = False
            ccVr = -1
            eta=np.linspace(0, 3, cmax+1)
            F1 = VKfield2D(loc, self.Dks, self.Dkd, np.exp(self.phaseSlip * 1j))
            while not(succ) and ccVr < cmax:
                ccVr += 1
                theta0 = np.random.rand(self.Dks.shape[0], self.Dkd.shape[1]) * 2 * np.pi
                theta0[0, 0] = 0.0
                Ephase = np.exp(theta0 * 1j)

                FI=VKfield2D(loc, self.Dks, self.Dkd, Ephase)

                succ = False
                cc=-1
                while not(succ) and cc<3:
                    cc=cc+1
                    Dvv = eta[ccVr]*F1*(-1)**cc+FI*(-1)**(cc//2)
                    Dvv[np.unravel_index(np.argsort(Dvv.flatten()), Dvv.shape)] = np.sort(self.D_Vr.rvs(size=Dvv.size))

                    Dvv = ZoDvR*ZoDv * (1 - Dvv)
                    Dvv[Dvv < limsVr[0]] = limsVr[0]
                    VrrF = Dvv * self.Vs

                    cmt = np.corrcoef(self.vpk.flatten(), VrrF.flatten())[0, 1]
                    if cmt > cVpVr[0] and cmt < cVpVr[1]:
                        self.Vr = VrrF
                        self.setOnsetTimes()
                        succ = np.sum(self.To + self.Trise * 1.2 > maxT) / self.To.size < 0.01
            succ = ccVr < cmax
        else: succ= False
        return succ

    def setRuptVelocHF(self, CVrr, stdVr, loc, phaseI, gbou=1000, Vfr=0.4, limsVr=None, maxT = 1E10,
                       cVpVr = None, cmax=4000):
        '''

        :param CVrr: float average ratio Vr/Vs
        :param stdVr: std of the ratio Vr/Vs in the rupture
        :param loc: [float, float, float] [lengtCorrelation strike, lengtCorrelation dip, 1+Hurts exponent] for Vr/Vs
        :param rnuc: radious for the nucleation area
        :param Vor: Vr/Vs deduction at the hypocenter
        :param gbou:  Gap to the borders where Vr decays
        :param Vfr: Vr/Vs deduction at the borders
        :param limsVr: [float, float] maximal and minimal Vr/Vs ratio
        :param maxT: maximal onset time +1.2Tr
        :param cVpVr: [float, float] Vr/Vs correlation range with vpeak
        :param cmax: maximal number of tries
        :return: if the function success
        '''

        if cVpVr is None: cVpVr = [0.2, 0.65]
        if limsVr is None: limsVr=[0.002, 1]
        #  Rupture propagation
        self.aVrr = CVrr  # Average ratio Vr/Vs
        self.locVr = loc

        ZoDv = np.ones(self.Vs.shape)
        res = self.Dll < gbou
        ZoDv[res] *= (1.0 - Vfr) * self.Dll[res] / gbou + Vfr
        res = self.Dww < gbou
        ZoDv[res] *= (1.0 - Vfr) * self.Dww[res] / gbou + Vfr
        res = self.nll[-1] - self.Dll < gbou
        ZoDv[res] *= (1.0 - Vfr) * (self.nll[-1] - self.Dll[res]) / gbou + Vfr
        res = self.nww[-1] - self.Dww < gbou
        ZoDv[res] *= (1.0 - Vfr) * (self.nww[-1] - self.Dww[res]) / gbou + Vfr

        #  Distribution 1-Vr/Vs
        coo = True
        ccVr = -2
        while coo and ccVr < cmax:
            resW = minimize(costWVr, np.asarray([1, 1]), args=(1-limsVr[1], CVrr, stdVr + 0.01 * (ccVr + 2)))
            coo = not (resW.success)
            ccVr += 1
        self.D_Vr = weibull_min(resW.x[0], loc=1-limsVr[1], scale=resW.x[1])

        succ = False
        ccVr = -1
        while not succ and ccVr < cmax:
            while not(succ) and ccVr < cmax:
                theta0 = np.random.rand(self.Dks.shape[0], self.Dkd.shape[1]) * 2 * np.pi
                theta0[0, 0] = 0.0
                phase = np.exp(theta0 * 1j)*(1 - self.F) + phaseI * self.F

                Dvv = VKfield2D(loc, self.Dks, self.Dkd, phase)
                Dvv *= -np.sign(np.corrcoef(self.vpk.flatten(), Dvv.flatten())[0, 1])
                Dvv[np.unravel_index(np.argsort(Dvv.flatten()), Dvv.shape)] = np.sort(self.D_Vr.rvs(size=Dvv.size))

                Dvv = ZoDv * (1 - Dvv)
                Dvv[Dvv < limsVr[0]] = limsVr[0]
                VrrF = Dvv * self.Vs

                cmt = np.corrcoef(self.vpk.flatten(), VrrF.flatten())[0, 1]
                succ = cmt > cVpVr[0] and cmt < cVpVr[1]
                ccVr += 1
            self.Vr = VrrF
            self.setOnsetTimes()
            succ = np.sum(self.To + self.Trise * 1.2 > maxT)/self.To.size<0.01
        succ = ccVr<cmax

        return succ

    def setSlipPattern(self, loc, tapPL_G=None, umax=None, randUmax=True):
        '''

        :param loc: [float, float, float] [lengtCorrelation strike, lengtCorrelation dip, 1+Hurts exponent]
        :param tapPL_G:  Tapper definition at each edge [-x,+x,-y,+y] default (None): [0.15, 0.15, 0.15, 0.15]
        :param umax: float, maximal slip, None it computes from a regresion on average slip
        :param vumax: float, variation of the regression for umax
        :return: if a Slip patter was solved
        '''
        if tapPL_G is None: tapPL_G = np.asarray([0.15, 0.15, 0.15, 0.15])

        self.cll = loc[0]
        self.cww = loc[1]
        self.H = loc[2]-1
        uni = self.Mo / np.sum(self.dll * self.dww * self.Rho * self.Vs ** 2)


        if umax is None: umax = computeUmax(uni, randUmax)[0]
        self.umax = umax

        ep = lambda uc: uni - truncexpon.mean(b=umax / uc, scale=uc)
        uc = fsolve(ep, 0.1 * uni)[0]
        self.D_u = truncexpon(b=umax / uc, scale=uc)

        self.tapPL = np.asarray([self.L * tapPL_G[0], self.L * tapPL_G[1], self.W * tapPL_G[2], self.W * tapPL_G[3]])
        # Tapper definition
        self.cdd = np.append(np.asarray(np.ceil(self.tapPL[:2] / self.dll).astype(int)),
                        np.asarray(np.ceil(self.tapPL[2:] / self.dll).astype(int)))
        self.ws = [np.hanning(2 * self.cdd[nd]) for nd in range(self.cdd.size)]

        Slip, succ, thethaS = computeSlip(self.Mo, self.Dks, self.Dkd, self.Vs, self.Rho, self.dll, self.dww,
                              loc, self.ade, self.D_u, self.cdd, self.ws)
        if succ:
            self.Slip = Slip
            self.phaseSlip = thethaS

        return succ

    def setSlipPatternHF(self, loc, phaseI, tapPL_G=None, umax=None, vumax=1.0):
        '''

        :param loc: [float, float, float] [lengtCorrelation strike, lengtCorrelation dip, 1+Hurts exponent]
        :param tapPL_G:  Tapper definition at each edge [-x,+x,-y,+y] default (None): [0.15, 0.15, 0.15, 0.15]
        :param umax: float, maximal slip, None it computes from a regresion on average slip
        :param vumax: float, variation of the regresion for umax
        :return: if a Slip patter was solved
        '''
        if tapPL_G is None: tapPL_G = np.asarray([0.15, 0.15, 0.15, 0.15])

        self.cll = loc[0]
        self.cww = loc[1]
        self.H = loc[2]-1
        uni = self.Mo / np.sum(self.dll * self.dww * self.Rho * self.Vs ** 2)

        if umax is None: umax = vumax*10 ** (0.95 * np.log10(uni) + 0.62)  # maximal slip

        if uni / umax > 0.4:
            umax = uni / 0.4
        self.umax = umax

        ep = lambda uc: uni - truncexpon.mean(b=umax / uc, scale=uc)
        uc = fsolve(ep, 0.1 * uni)[0]
        self.D_u = truncexpon(b=umax / uc, scale=uc)

        self.tapPL = np.asarray([self.L * tapPL_G[0], self.L * tapPL_G[1], self.W * tapPL_G[2], self.W * tapPL_G[3]])
        # Tapper definition
        self.cdd = np.append(np.asarray(np.ceil(self.tapPL[:2] / self.dll).astype(int)),
                        np.asarray(np.ceil(self.tapPL[2:] / self.dll).astype(int)))
        self.ws = [np.hanning(2 * self.cdd[nd]) for nd in range(self.cdd.size)]

        Slip, succ, phaseS = addHFSlip(self.Mo, self.Dks, self.Dkd, self.Vs, self.Rho, self.dll, self.dww,
                              loc, self.ade, self.D_u, self.cdd, self.ws, phaseI, self.F, 0.6)
        if succ:
            self.Slip = Slip
            self.phaseSlip = phaseS

        return succ

    def writePickle(self, filS):
        self.file_name=filS
        with open(self.file_name, "wb") as file_handle:
            pickle.dump(self, file_handle)
            file_handle.close()
