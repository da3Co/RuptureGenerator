import numpy as np
import RuptG as rg
from os import path, makedirs


Mw = 7.0  # Magnitude
L = 40*1E3  # Length in the strike direction
W = 25*1E3  # Width in the dip direction
strike, dip, Arake = 75, 65, 100  # Strike, Dip, and average rake respectively
dll, dww = 200, 200  # Size in strike (dll) and dip (dww) directions of the sub-faults
PoiR_I = np.asarray([0.5, 1.0])  # Relative location of a fix point (reference the left down corner)
PoiL = np.asarray([0, 0, -3.2E3])  # Absolute location of PoiR
VsU=3200 # Uniform Shear velocity (Vs) for the fault
RhoU=2750 # Uniform Density (Rho) for the fault
CVrr_I = 0.75  # Average rupture velocity coefficient with respect Vs
stdVr = 0.25  # Std rupture velocity coefficient with respect Vs

Sty = 'SD'  # 'SD'#'SS' #--> SS: Strike-Slip SD:Slip-Dip
SuD = 'SU'  # 'SS'#'CR' # SU #--> SS:'Strike-Slip' CR: Crustal-Dip SLip SU: Subduction-Dip Slip
regTr='Som'  # Regression for average rise time
KtacM = 0.05  # Tpeak average
dtt=1E-2  # Step time
foldS='Results/Example1/'
name='Resu_1'
#  ----------------------- End Inputs ------------------------------------------------------------------------------
if not path.exists('%s' % foldS): makedirs(foldS[:-1])

Mo = 10 ** (Mw * 1.5 + 9.05)
RR = rg.Rupture(Mo, L, W)

RR.assingLocationsOrientations(strike, dip, Arake, dll, dww, PoiR_I, PoiL)
RR.assingMaterialProp(VsU*np.ones(RR.Dll.shape), RhoU*np.ones(RR.Dll.shape))

loc = np.append(rg.computeVKParams(np.asarray([L]), np.asarray([W]), Mw, Sty), [1.75])
sucS = RR.setSlipPattern(loc)
if sucS:
    #  Rakes variation definition
    RR.setRakesVariations()

    #  Rise Time definition
    trp_I, strp = rg.computeAveRiseTime(regTr, Mw)
    sucTr = RR.setRiseTimePattern(loc, KtacM, trp_I)

    if sucTr:
        #  Hypocenter location
        RR.setHypocenter(Sty, SuD)

        #  Rupture velocity pattern and define Onset times
        locVr = [loc[0], loc[1], 0.7]
        sucV = RR.setRuptVeloc(CVrr_I, stdVr, locVr)

        if sucV:
            #  Compute STF at each subfault
            RR.generateSTFOpt(dtt)

            #  Write sources
            RR.writePickle(foldS + '%s.pickle' % name)
        else:
            print('No success in rupture velocity creation')
    else:
        print('No success in rise time creation')

else:
    print('No success in slip creation')

print('Fin')