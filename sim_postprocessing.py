"""
Dafne Project - Simulation Processing Toolbox
=========================
:title:     Simulation post-processing
:authors:   MarcoMicotti@polimi
:content:   Post processing of WEF simulations to compute energy production and fish yield in the Omo-Turkana case study
"""


import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pdb
from pathlib import Path

#import custom modules
sys.path.insert(0,'PATH/TO/DAFNE/TOOLBOX/HERE/' )
import dafneutils as du
import indicatorslib as il
import DAF_importer
import globalparam as gp
import importlib
importlib.reload(du)
importlib.reload(gp)
importlib.reload(il)
importlib.reload(DAF_importer)

################################################################################
# init & load data and parameters
################################################################################
TSPATH      = gp.simDOtb
tOtb        = pd.date_range(gp.dafneParam['OTB']['tStart'], gp.dafneParam['OTB']['tEnd'],freq = 'D')
tOtbY       = pd.date_range(gp.dafneParam['OTB']['tStart'], gp.dafneParam['OTB']['tEnd'],freq = 'Y')
pathwLabels = gp.dafneParam['OTB']['pwLabels']

#fish biomass parameters
alfa  = gp.dafneParam['OTB']['FishTurkana']['alfa']
beta  = gp.dafneParam['OTB']['FishTurkana']['beta']
gamma = gp.dafneParam['OTB']['FishTurkana']['gamma']

#computation
computeEnergy = True
computeFish   = True

#reservoir quick dictionary, to be move to globalparam
resDict     = { 'G1'    : {'label': 'Gibe_I', 'min_ol': 1653, 'head_min': 207.3,'MEF':1.3},
                'G2'    : {'label': 'Gibe_II'},
                'G3'    : {'label': 'Gibe_III','min_ol': 800,'head_min':186},
                'KOY'   : {'label': 'Koysha','min_ol' : 616, 'head_min':150 }
                }


################################################################################
# functions
################################################################################

def compute_energy(res, level, flow):
    '''given reservoir level and flow, compute energy production'''
    if res == 'G2':
        head = level
    else:
        head =  np.maximum(level,resDict[r]['min_ol']) - resDict[r]['min_ol'] + resDict[r]['head_min']
    q_turb  = np.minimum(flow,gp.dafneParam['OTB']['qmax_hyd'][r])
    en      = q_turb*head*gp.dafneParam['OTB']['eff_hyd']*9.81*1000*24/1000000    #in MWh
    return en,q_turb

def get_amplitude(level):
    '''return amplitude of level oscillation between dry season (January to August) and wet season (September to December)'''
    idx_dry     = (level.index.month >= 1 ) & (level.index.month < 9 )
    idx_wet     = (level.index.month >= 9 ) & (level.index.month <= 12 )
    level_dry   = level.loc[idx_dry,:]
    level_wet   = level.loc[idx_wet,:]
    dry_max     = level_dry.groupby(level_dry.index.year).max()
    wet_max     = level_wet.groupby(level_wet.index.year).max()
    amplitude   = wet_max - dry_max
    return amplitude

def compute_fishYield(level):
    '''Fish biomass from lake Turkana levels'''
    ampl        = get_amplitude(level)
    h_Y_Mean    = level.groupby(level.index.year).mean()
    yy          = h_Y_Mean.index
    #note -> cannot use df timeindex because of formulation, including h(t-1)
    fishYield   = alfa*h_Y_Mean.iloc[0:-1,:].values + beta*ampl.iloc[1:,:].values + gamma
    fishYieldDf = pd.DataFrame(fishYield, columns=h_Y_Mean.columns, index=h_Y_Mean.index[1:])
    return fishYieldDf


################################################################################
# processing
################################################################################

###### energy production
if computeEnergy:
    print('computing Energy production')
    for r in resDict:
        if r == 'G2':
            #quick&dirty workaround for G2
            qG1   = pd.read_csv(os.path.join(TSPATH,'OTB_RCP45_R17_Gibe_I_95f0fc8_Outflow.csv'), sep=',', header=0)
            level = qG1.copy()
            level.loc[:,:] = gp.dafneParam['OTB']['h_G2'] #see tesi di Badagliacca e Spinelli
            flow  = np.maximum((qG1 - resDict['G1']['MEF']),0)
        else:
            h_file      = os.path.join(TSPATH,'OTB_RCP45_R17_{}_95f0fc8_Level.csv'.format(resDict[r]['label']))
            q_file      = os.path.join(TSPATH,'OTB_RCP45_R17_{}_95f0fc8_Outflow.csv'.format(resDict[r]['label']))
            level       = pd.read_csv(h_file, sep=',', header=0)
            flow        = pd.read_csv(q_file, sep=',', header=0)
        #
        if 'timestart' in level.columns: level.drop(columns='timestart',inplace=True)
        # level.columns= pathwLabels
        level.set_index(tOtb,inplace = True)
        if 'timestart' in flow.columns: flow.drop(columns='timestart',inplace=True)
        # flow.columns= pathwLabels
        flow.set_index(tOtb,inplace = True)

        en,qturb = compute_energy(r,level,flow)
        en.to_csv(os.path.join(TSPATH,'en45_{}.csv'.format(r)),index=False)
        qturb.to_csv(os.path.join(TSPATH,'qturb45_{}.csv'.format(r)),index=False)
        print('{} reservoir and hpp processed'.format(r))

########fish yield
if computeFish:
    print('computing Fish Yield in the Turkana lake')
    hTur      = pd.read_csv(os.path.join(TSPATH,'OTB_RCP45_R17_Turkana_95f0fc8_Level.csv'),sep=',', header=0)
    if 'timestart' in hTur.columns: hTur.drop(columns='timestart',inplace=True)
    # hTur.columns= pathwLabels
    hTur.set_index(tOtb,inplace = True)
    # import pdb; pdb.set_trace()
    fishYield       = compute_fishYield(hTur)
    # fishYieldHist   = compute_fishYield(hTurHist)
    # fishDeficit     = (np.nanmean(fishYieldHist.values) - fishYield).abs()
    fishYieldHist   = gp.dafneParam['OTB']['TurFishNat']
    fishDeficit     = (fishYieldHist - fishYield).abs()
    fishYield.to_csv(os.path.join(TSPATH,'fishYield_Turkana_45.csv'),index=False)
    fishDeficit.to_csv(os.path.join(TSPATH,'fishYieldDeficit_Turkana_45.csv'),index=False)
    print('computing Fish Yield in the Turkana lake: done!')
