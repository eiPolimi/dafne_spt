'''
Dafne Project - Simulation Processing Toolbox
=========================
:title: DAFNE Global Parameters
:authors: MarcoMicotti@polimi
:content: declares a dictionary of input/output path and of useful parameters, imported by the other script of the toolbox
:notes: updated with 2020 parameters.
'''
import os
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
import pdb

################################################################################
# init & load data and parameters
################################################################################

# set paths
wd          = '/media/marco/Samsung_T3/work/dei/DAFNE/WEF_output/'
wdDaf       = '/media/marco/Samsung_T3/work/dei/DAFNE/DAF_output/'
nslDir      = 'NSL2020'
simDir      = 'simulation_results'
csvTsDir    = 'timeseries'
csvDashDir  = 'dashboard'

# set scenarios label
scenLabel   = 'RCP45_R17'
scenotb     = 'otbfinalnslz'
scenzrb     = 'zrbsummersc'

# other global variables
geop_no_value  = -999.9
runDebug    = True
authorName  = 'polimi@dafneproject'
origin      = 'DAFNE project: DAF optimization 2nd run'
origin      = 'DAFNE project: WEF integrated model simulation'


#hV relationships as the system model (attached, where ID=1,3,6,7 matches Name=Gibe1,Gibe3,Koysha,Lake_Turkana)
hV_curve        = pd.read_csv(os.path.join(wd,'OTB',nslDir,'param','Omo_curve_hV.dat'),sep='\s+')
G3_lsv          = pd.read_csv(os.path.join(wd,'OTB/NSL2020/param/lsv_GibeIII.txt'),sep='\s+',header=None)
G3_lsv.index    = ['h','s','v']
# import pdb; pdb.set_trace()
KOY_lsv         = pd.read_csv(os.path.join(wd,'OTB/NSL2020/param/lsv_Koysha.txt'),sep='\s+',header=None)
KOY_lsv.index   = ['h','s','v']
h0G3            = 660
h0KOY           = 510

dafneParam = {
            'ZRB' : {'wd'      : os.path.join(wdDaf,'ZRB',nslDir),
                    'simD'     : os.path.join(wdDaf,'ZRB',nslDir,simDir),
                    'pcDir'    : os.path.join(wdDaf,'ZRB',nslDir,'parcoord',scenzrb),
                    'tsDir'    : os.path.join(wdDaf,'ZRB',nslDir,csvTsDir,scenzrb),
                    'dashDir'  : os.path.join(wdDaf,'ZRB',nslDir,csvDashDir,scenzrb),
                    'imgDir'   : os.path.join(wdDaf,'ZRB',nslDir,'img',scenzrb),
                    'objFile'  : os.path.join(wdDaf,'ZRB',nslDir,'Zambezi_new_targets.reference'),
                    'scenLabel': scenLabel,
                    'scen'     : scenzrb,
                    # 'pwLabels' : ['P.21','P.51''P.40','P.6','P.73','P.0','P.62'], #mapping wrt to the full set of 2020 january solutions
                    'pwLabels' : ['P1','P2','P3','P4','P5','P6','P7'],
                    'freq'     : 'monthly',
                    'tStart'   : dt.datetime(2020, 1, 1),
                    'tEnd'     : dt.datetime(2059, 12, 31),
                    'timestep' : 'M',
                    'indUrl'   : 'http://xake.deib.polimi.it:8081/drupal/zambezi/?q=indicators/'
                    'origin'   : 'DAFNE project: DAF optimization 2nd run'
                    },
            'OTB' : {'wd'      : os.path.join(wd,'OTB',nslDir),
                    'simD'     : os.path.join(wd,'OTB',nslDir,simDir,'csv'),
                    'pcDir'    : os.path.join(wd,'OTB',nslDir,'parcoord',scenotb),
                    'tsDir'    : os.path.join(wd,'OTB',nslDir,csvTsDir,scenotb),
                    'dashDir'  : os.path.join(wd,'OTB',nslDir,csvDashDir,scenotb),
                    'imgDir'   : os.path.join(wd,'OTB',nslDir,'img',scenotb),
                    'pklDir'   : os.path.join(wd,'OTB',nslDir,'pyObjects',scenotb),
                    'refDir'   : os.path.join(wd,'OTB',nslDir,'references'),
                    'simCode'  : '95f0fc8', #WEF model simulation code, last run
                    'scenLabel': scenLabel,
                    'scen'     : scenotb,
                    # 'pwLabels' : ['P3.470','P3.330','P3.607','P3.428','P3.394','P3.89'], #pathw selection done during 2019 NSL
                    # 'pwLabels' : ['P153','P161','P452','P519','P565','P594'], #new labels taken from updated simulation
                    'pwLabels' : ['P5','P2','P6','P4','P3','P1'], #pathw selection done during 2019 NSL, renamed
                    'freq'     : 'daily',
                    'tStart'   : dt.datetime(2021, 1, 1),
                    'tEnd'     : dt.datetime(2099, 12, 31),
                    'timestep' : 'D',
                    'indUrl'   : 'http://xake.deib.polimi.it:8081/drupal/omo-turkana/?q=indicators/',
                    'origin'            : 'DAFNE project: WEF integrated model simulation'
                    'TurFishNat'        : 7.9977,
                    'G3_lev_vol_f'      : {'level': hV_curve.loc[hV_curve['ID'] == 3,'h'].values, 'volume': hV_curve.loc[hV_curve['ID'] == 3,'V'].values},
                    'KOY_lev_vol_f'     : {'level': hV_curve.loc[hV_curve['ID'] == 6,'h'].values, 'volume': hV_curve.loc[hV_curve['ID'] == 6,'V'].values},
                    'G3_lev_surf_f'     : {'level': G3_lsv.loc['h',:].values + h0G3, 'surface': G3_lsv.loc['s',:].values / 10**6}, #surfaces in km2
                    'KOY_lev_vol_f'     : {'level': hV_curve.loc[hV_curve['ID'] == 6,'h'].values, 'volume': hV_curve.loc[hV_curve['ID'] == 6,'V'].values},
                    'KOY_lev_surf_f'    : {'level': KOY_lsv.loc['h',:].values + h0KOY, 'surface': KOY_lsv.loc['s',:].values / 10**6}, #surfaces in km2
                    'FishTurkana'       : {'alfa' : 0.3252,'beta':1.006, 'gamma': -110.91}, #see pg 30 deliverable D52, (Gownaris et al., 2017)
                    'FishInRes'         : {'coeff' : 8.32,'exp':0.92}, #see pg 17 deliverable D54
                    'IR1'               : {'Nominal_WDemand'      : 180, 'Area' : 790},  # wd: m3/sec, Area expressed in km2, from the model configuration file
                    'IR2'               : {'Nominal_WDemand'      : 34, 'Area' : 314},   # wd: m3/sec, Area expressed in km2, from the model configuration file
                    'qmax_hyd'          : {'G1': 3*34, 'G2': 4*25.4, 'G3': 10*95, 'KOY': 8*192},
                    'h_G2'              : 505,
                    'eff_hyd'           : 0.8,
                    'g_hyd'             : 9.81/1000
                }
            }


refLevels = {
            'ITT': {'max_ol':1030.5 ,'min_ol':1006},
            'KGU': {'max_ol':977 ,'min_ol':972.3}, #originally was 975.4. 972.3 m according to Cervigni et al. (2015)
            'KGL': {'max_ol':582 ,'min_ol':530},
            'DEV': {'max_ol':595 ,'min_ol':np.nan},
            'BAT': {'max_ol':762 ,'min_ol':746},
            'KAR': {'max_ol':488.5 ,'min_ol':475.5},
            'CAB': {'max_ol':326 ,'min_ol':295},
            'MNK': {'max_ol':207 ,'min_ol':np.nan},
            'G3': {'max_ol':892 ,'min_ol':800}, #abs taken from comparison among WEF & DAF level/Volume curves
            'KOY': {'max_ol':680 ,'min_ol':510}} #abs taken from comparison among WEF & DAF level/Volume curves

###create non existing dir

for ws in dafneParam:
    for k in dafneParam[ws]:
        d = dafneParam[ws][k]
        if isinstance(d,str):
            if not(os.path.exists(d)):
                # if runDebug: pdb.set_trace()
                Path(d).mkdir(parents=True, exist_ok=True)


###ZRB shortcut

wdZrb       = dafneParam['ZRB']['wd']
scenarioZrb = dafneParam['ZRB']['scen']
simDZrb     = dafneParam['ZRB']['simD']
pcDirZrb    = dafneParam['ZRB']['pcDir']
freqZrb     = dafneParam['ZRB']['freq']

###OTB shortcut
wdOtb       = dafneParam['OTB']['wd']
scenarioOtb = dafneParam['OTB']['scen']
simDOtb     = dafneParam['OTB']['simD']
pcDirOtb    = dafneParam['OTB']['pcDir']
freqOtb     = dafneParam['OTB']['freq']
