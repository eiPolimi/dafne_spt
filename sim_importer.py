'''
Dafne Project - Simulation Processing Toolbox
=========================
:title:     Simulation import
:authors:   MarcoMicotti@polimi
:content:   imports the outcomes of DAF/WEF simulations into a python object (.pkl) including all the components time series and saving it into the file system.
            Different output formats of DAF/WEF simulations are supported by setting up customized dictionary within the script;
'''

import os
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import pdb
import sys
sys.path.insert(0,'PATH/TO/DAFNE/TOOLBOX/HERE/' )
import dafneutils as du
import globalparam as gp
import indicatorslib as il
import importlib
importlib.reload(du)

################################################################################
# init
################################################################################
###init system models
zrb         = du.initZrb()
otb         = du.initOtb()
###time
#Zambezi: time vector with monthly start frequency
tZrb        = pd.date_range(gp.dafneParam['ZRB']['tStart'], gp.dafneParam['ZRB']['tEnd'],freq = 'MS')
#Omo: daily time vector
tOtb        = pd.date_range(gp.dafneParam['OTB']['tStart'], gp.dafneParam['OTB']['tEnd'],freq = 'D')
#Omo: yearly time vector for fish yield. First year is removed because it is derived from previous year water level average.
tOtbY       = pd.date_range(gp.dafneParam['OTB']['tStart'], gp.dafneParam['OTB']['tEnd'],freq = 'YS')[1:]

#Zambezi features dictionary
zrbDict  = {
            'VIC' :   { 'Power' : 108, 'Status': 'existing'},
            'KAR' :   { 'Power' : 1830, 'Status': 'existing'},
            'ITT' :   { 'Power' : 120, 'Status': 'existing'},
            'KGU' :   { 'Power' : 990, 'Status': 'existing'},
            'KGL' :   { 'Power' : 750, 'Status': 'existing'},
            'CAB' :   { 'Power' : 2075, 'Status': 'existing'},
            'BAT' :   { 'Power' : 1600, 'Status': 'planned'},
            'DEV' :   { 'Power' : 1240, 'Status': 'planned'},
            'MNK' :   { 'Power' : 1300, 'Status': 'planned'}
            }

#Omo-Turkana candidate pathways dictionary
otbDict  = {'baseline' :   {
                                'shortName' : 'B',
                                'labelPre'  : 'P0.',
                                'fileName'  : 'base_obj.txt',
                                'indLabels' : ['j_Hyd','j_Env','j_Rec','j_Fish']
                            },
            'koysha' :   {
                                'shortName' : 'K',
                                'labelPre'  : 'P1.',
                                'fileName'  : 'koysha_obj.txt',
                                'indLabels' : ['j_Hyd','j_Env','j_Rec','j_Fish']
                            },
            'irrigation' :   {
                                'shortName' : 'I',
                                'labelPre'  : 'P2.',
                                'fileName'  : 'irr_obj.txt',
                                'indLabels' : ['j_Hyd','j_Env','j_Rec','j_Fish','j_Irr']
                            },
            'irrKoysha' :   {
                                'shortName' : 'IK',
                                'labelPre'  : 'P3.',
                                'fileName'  : 'irrKoysha_obj.txt',
                                'indLabels' : ['j_Hyd','j_Env','j_Rec','j_Fish','j_Irr']
                            }
            }

###
runDebug    = True
# runDebug    = False

################################################################################
# functions
################################################################################
def datetime2matlabdn(dtime):
    '''taken from: https://stackoverflow.com/questions/8776414/python-datetime-to-matlab-datenum#8776555'''
    ord = dtime.toordinal()
    mdn = dtime + dt.timedelta(days = 366)
    frac = (dtime - dt.datetime(dtime.year,dtime.month,dtime.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
    return mdn.toordinal() + frac

def zrbDataFolderDict(pathwayLabel):
    """returns a dictionary where the keys are the filenames of the simulation output folder, for the Zambezi River Basin"""
     # pathNum     = pathwayLabel.replace('P','')
    pathNum     = pathwayLabel
    fileDict    = {
            '{}_bg.txt'.format(pathNum): {'component' : 'BAT', 'varList': ['a', 'h','s','s1','r','u'],'nanValues':{'h':0,'s':0,'s1':0}},
            '{}_cb.txt'.format(pathNum): {'component' : 'CAB', 'varList': ['a', 'h','s','s1','r','u'],'nanValues':{'h':0,'s':0,'s1':0}},
            '{}_dg.txt'.format(pathNum): {'component' : 'DEV', 'varList': ['a', 'h','s','s1','r','u'],'nanValues':{'h':0,'s':0,'s1':0}},
            '{}_itt.txt'.format(pathNum): {'component' : 'ITT', 'varList': ['a', 'h','s','s1','r','u'],'nanValues':{'h':0,'s':0,'s1':0}},
            '{}_ka.txt'.format(pathNum): {'component' : 'KAR', 'varList': ['a', 'h','s','s1','r','u'],'nanValues':{'h':0,'s':0,'s1':0}},
            '{}_kgl.txt'.format(pathNum): {'component' : 'KGL', 'varList': ['a', 'h','s','s1','r','u'],'nanValues':{'h':0,'s':0,'s1':0}},
            '{}_kgu.txt'.format(pathNum): {'component' : 'KGU', 'varList': ['a', 'h','s','s1','r','u'],'nanValues':{'h':0,'s':0,'s1':0}},
            '{}_mn.txt'.format(pathNum): {'component' : 'MNK', 'varList': ['a', 'h','s','s1','r','u'],'nanValues':{'h':0,'s':0,'s1':0}},
            '{}_hp_bg.txt'.format(pathNum): {'component' : 'BAT', 'varList': ['tr','en_twh','ed_twh'],'nanValues':{'tr':0,'en_twh':0}},
            '{}_hp_cb.txt'.format(pathNum): {'component' : 'CAB', 'varList': ['tr','en_twh','ed_twh'],'nanValues':{'tr':0,'en_twh':0}},
            '{}_hp_dg.txt'.format(pathNum): {'component' : 'DEV', 'varList': ['tr','en_twh','ed_twh'],'nanValues':{'tr':0,'en_twh':0}},
            '{}_hp_itt.txt'.format(pathNum): {'component' : 'ITT', 'varList': ['tr','en_twh','ed_twh'],'nanValues':{'tr':0,'en_twh':0}},
            '{}_hp_ka.txt'.format(pathNum): {'component' : 'KAR', 'varList': ['tr','en_twh','ed_twh'],'nanValues':{'tr':0,'en_twh':0}},
            '{}_hp_kgl.txt'.format(pathNum): {'component' : 'KGL', 'varList': ['tr','en_twh','ed_twh'],'nanValues':{'tr':0,'en_twh':0}},
            '{}_hp_kgu.txt'.format(pathNum): {'component' : 'KGU', 'varList': ['tr','en_twh','ed_twh'],'nanValues':{'tr':0,'en_twh':0}},
            '{}_hp_mn.txt'.format(pathNum): {'component' : 'MNK', 'varList': ['tr','en_twh','ed_twh'],'nanValues':{'tr':0,'en_twh':0}},
            '{}_irr2.txt'.format(pathNum): {'component' : 'ID2', 'varList': ['wa','wd'],'nanValues':{}},
            '{}_irr3.txt'.format(pathNum): {'component' : 'ID3', 'varList': ['wa','wd'],'nanValues':{}},
            '{}_irr4.txt'.format(pathNum): {'component' : 'ID4', 'varList': ['wa','wd'],'nanValues':{}},
            '{}_irr5.txt'.format(pathNum): {'component' : 'ID5', 'varList': ['wa','wd'],'nanValues':{}},
            '{}_irr6.txt'.format(pathNum): {'component' : 'ID6', 'varList': ['wa','wd'],'nanValues':{}},
            '{}_irr7.txt'.format(pathNum): {'component' : 'ID7', 'varList': ['wa','wd'],'nanValues':{}},
            '{}_irr8.txt'.format(pathNum): {'component' : 'ID8', 'varList': ['wa','wd'],'nanValues':{}},
            '{}_irr9.txt'.format(pathNum): {'component' : 'ID9', 'varList': ['wa','wd'],'nanValues':{}},
            # '{}_simulated.objs'.format(pathNum): {'component' : 'zrb', },
            '{}_timing_bg.txt'.format(pathNum): {'component' : 'BAT', 'featList': ['Power'],'nanValues':{'Power':0}},
            '{}_timing_dg.txt'.format(pathNum): {'component' : 'DEV', 'featList': ['Power'],'nanValues':{'Power':0}},
            '{}_timing_mn.txt'.format(pathNum): {'component' : 'MNK', 'featList': ['Power'],'nanValues':{'Power':0}},
            # '{}_Victoria.txt'.format(pathNum): {,
            '{}_Delta.txt'.format(pathNum): {'component': 'ZDE', 'varList': ['q','tf'],'nanValues':{}}
            }
    return fileDict
#

def otbDataFolderDict(scenLabel):
    """returns a dictionary where the keys are the filenames of the simulation output folder, for the Omo-Turkana Basin"""
    simCode     = gp.dafneParam['OTB']['simCode']
    scenNum     = scenLabel.split('_')[0][-2:]
    fileDict    = {
            'en{}_G1.csv'.format(scenNum)                              : {'component':'G1' ,'varLabel':'en_gwh','separator': ',','header':0},#check unit!!
            'qturb{}_G1.csv'.format(scenNum)                           : {'component':'G1' ,'varLabel':'tr','separator': ',','header':0},
            'en{}_G2.csv'.format(scenNum)                              : {'component':'G2' ,'varLabel':'en_gwh','separator': ',','header':0},#check unit!!
            'qturb{}_G2.csv'.format(scenNum)                           : {'component':'G2' ,'varLabel':'tr','separator': ',','header':0},
            'en{}_G3.csv'.format(scenNum)                              : {'component':'G3' ,'varLabel':'en_gwh','separator': ',','header':0},#check unit!!
            'qturb{}_G3.csv'.format(scenNum)                           : {'component':'G3' ,'varLabel':'tr','separator': ',','header':0},
            'en{}_KOY.csv'.format(scenNum)                             : {'component':'KOY' ,'varLabel':'en_gwh','separator': ',','header':0},#check unit!!
            'qturb{}_KOY.csv'.format(scenNum)                          : {'component':'KOY' ,'varLabel':'tr','separator': ',','header':0},
            'OTB_{}_Gibe_I_{}_Inflow.csv'.format(scenLabel,simCode)    : {'component':'G1' ,'varLabel':'a','separator': ',','header':0},
            'OTB_{}_Gibe_I_{}_Level.csv'.format(scenLabel,simCode)     : {'component':'G1' ,'varLabel':'h','separator': ',','header':0},
            'OTB_{}_Gibe_I_{}_Outflow.csv'.format(scenLabel,simCode)   : {'component':'G1' ,'varLabel':'r','separator': ',','header':0},
            'OTB_{}_Gibe_I_{}_Volume.csv'.format(scenLabel,simCode)    : {'component':'G1' ,'varLabel':'s','separator': ',','header':0},
            'OTB_{}_Gibe_III_{}_Inflow.csv'.format(scenLabel,simCode)  : {'component':'G3' ,'varLabel':'a','separator': ',','header':0},
            'OTB_{}_Gibe_III_{}_Level.csv'.format(scenLabel,simCode)   : {'component':'G3' ,'varLabel':'h','separator': ',','header':0},
            'OTB_{}_Gibe_III_{}_Outflow.csv'.format(scenLabel,simCode) : {'component':'G3' ,'varLabel':'r','separator': ',','header':0},
            'OTB_{}_Gibe_III_{}_Volume.csv'.format(scenLabel,simCode)  : {'component':'G3' ,'varLabel':'s','separator': ',','header':0},
            'OTB_{}_Koysha_{}_Inflow.csv'.format(scenLabel,simCode)    : {'component':'KOY' ,'varLabel':'a','separator': ',','header':0},
            'OTB_{}_Koysha_{}_Level.csv'.format(scenLabel,simCode)     : {'component':'KOY' ,'varLabel':'h','separator': ',','header':0},
            'OTB_{}_Koysha_{}_Outflow.csv'.format(scenLabel,simCode)   : {'component':'KOY' ,'varLabel':'r','separator': ',','header':0},
            'OTB_{}_Koysha_{}_Volume.csv'.format(scenLabel,simCode)    : {'component':'KOY' ,'varLabel':'s','separator': ',','header':0},
            'OTB_{}_Turkana_delta_{}_QChan.csv'.format(scenLabel,simCode): {'component':'ODE' ,'varLabel':'q','separator': ',','header':0},
            'OTB_{}_Turkana_{}_Level.csv'.format(scenLabel,simCode)    : {'component':'TUR','varLabel':'h','separator': ',','header':0},
            'fishYield_Turkana_{}.csv'.format(scenNum)                      : {'component':'TUR','varLabel':'fy','separator': ',','header':0},
            'fishYieldDeficit_Turkana_{}.csv'.format(scenNum)               : {'component':'TUR','varLabel':'fd','separator': ',','header':0},
            'OTB_{}_Kuraz_{}_water_irrigat.csv'.format(scenLabel,simCode)   : {'component': 'IR1','varLabel':'wa','separator': ',','header':0},
            'OTB_{}_Private_{}_water_irrigat.csv'.format(scenLabel,simCode) : {'component': 'IR2','varLabel':'wa','separator': ',','header':0},
            'OTB_{}_Lower_Omo_recession_{}_QChan.csv'.format(scenLabel,simCode)  : {'component': 'REC','varLabel':'q','separator': ',','header':0},
            }
    return fileDict
#

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output)

def importVariables(comp,dataDf,pathwLabel):
    '''loop on variables and import into the component'''
    for v in dataDf.columns:
        # v    = data.columns[0]
        try:
            idxv = [cv.label for cv in comp.variables].index(v)
            compV= comp.variables[idxv]
        #loop on alternatives ??
            compV.data[pathwLabel] = dataDf[v]
        #end of loop
        except:
            print('{} - {}: missing from variable list'.format(comp.name,v))
            continue
    return comp

def importFeatures(comp,dataDf,pathwLabel):
    '''loop on features and import into the component'''
    for f in dataDf.columns:
        # f    = data.columns[0]
        try:
            idxf = [cf.label for cf in comp.features].index(f)
            comp.features[idxf].data[pathwLabel] = dataDf[f]
        #end of loop
        except:
            print('{} - {}: missing from features list'.format(comp.name,f))
            continue
    return comp

def addTimeIndex(df,tIndex): #former add29Feb
    '''add daily time index to dataframe using standard 365days-long years, adding 29 of February where necessary with the same value of Feb 28'''
    idxLeap = (tIndex.day == 29 ) & (tIndex.month == 2 )
    df.set_index(tIndex[~(idxLeap)],inplace = True)
    dfTmp   = pd.DataFrame(index=tIndex)
    df      = pd.concat([dfTmp,df],axis=1)
    idx28   = np.where(idxLeap)[0]-1
    df.iloc[idxLeap,:] = df.iloc[idx28,:].to_numpy()
    return df

def cycloRef2Df(fileName,time,refLabel,transpose=False):
    """create a dataframe with the simulation time horizon, repeating the cyclo-stationary reference time series given in the fileName"""
    if transpose:
        dfRef   = pd.read_csv(fileName,sep='\t',header=None).T
    else:
        dfRef   = pd.read_csv(fileName,sep='\t',header=None)
    dfRef.index = range(1,len(dfRef)+1)
    dfRef.columns = ['ref']
    dfOut   = pd.DataFrame(time,columns=['time'])
    dfOut.set_index('time',drop=False,inplace=True)
    dfOut.set_index('time',inplace=True)
    dfOut['doy'] = dfOut.index.dayofyear
    dfTmp   = dfRef.reindex(index= dfOut['doy'].to_list())
    dfTmp.index = time
    # dfOut   = pd.concat([dfOut, ,:]],axis=1,ignore_index=True)
    dfOut[refLabel] = dfTmp
    dfOut.loc[dfOut['doy']==366,refLabel] = dfRef.loc[365].to_list()
    return dfOut

def applyCoeff(df, colLabel, coeff, operator):
    """perform operator with a coefficient on selected columns of df dataframe"""
    if colLabel:
        idx         = colLabel
    else:
        idx         = df.columns
    exp2Eval      = 'df.loc[:,idx] {} coeff'.format(operator)
    df.loc[:,idx] = eval(exp2Eval)
    return df

def identifyPathwGroup(watersystem, simDir):
    '''starting from timing information, identify labels for pathways group returning a dataframe with old labels as index and newLabel as column'''
    # resDict = {'BAT':'bg','DEV':'dg','MNK':'mn'}
    timeDf  = pd.DataFrame()
    for p in os.listdir(simDir):
        pDict = zrbDataFolderDict(p)
        fileL = os.listdir(os.path.join(simDir,p))
        for f in [tf for tf in fileL if 'timing' in tf]:
            if f in pDict.keys():
                try:
                    df      = pd.read_csv(os.path.join(simDir,p,f),sep=' ')
                    timeDf.loc[pDict[f]['component'],p] = df.min().to_numpy()
                except:
                    timeDf.loc[pDict[f]['component'],p] = np.nan
                # continue
    newLabelDf = pd.DataFrame(timeDf.isna().sum())
    newLabelDf.loc[newLabelDf[0] == 0,'nRes'] = '3'
    newLabelDf.loc[newLabelDf[0] == 1,'nRes'] = '2'
    newLabelDf.loc[newLabelDf[0] == 2,'nRes'] = '1'
    newLabelDf.loc[newLabelDf[0] == 3,'nRes'] = '0'
    # if runDebug: pdb.set_trace()
    newLabelDf['newLabel']  =  'P' + newLabelDf['nRes'] + '.' + newLabelDf.index.str.replace('P','')
    newLabelDf.drop(columns=0,inplace=True)
    # if runDebug: pdb.set_trace()
    newLabelDf.to_csv(os.path.join(gp.dafneParam[watersystem.acronym]['wd'],'newlabels.csv'))
    return newLabelDf

################################################################################
# load data functions
################################################################################

def loadZrbData(watersystem):
    '''loop over pathways folder and import into the watersystem object - Zambezi river basin'''
    pDir        = os.path.join(gp.simDZrb)
    tmpL        = []
    #loop on pathways classes or model configuration -> 1 folder each
    for p in os.listdir(pDir):
        if os.path.isfile(os.path.join(pDir,p)): continue
        pNewLab     = p
        fileList    = os.listdir(os.path.join(pDir,p))
        dataD       = zrbDataFolderDict(p)
        #loop on files in pathw folder
        for f in fileList:
            if 'timing' in f:
                compAcr     = dataD[f]['component']
                feat        = dataD[f]['featList']
                try:
                    dataM       = pd.read_csv(os.path.join(pDir,p,f), sep=' ')
                    dataM.replace(dataD[f]['nanValues'],np.nan,inplace=True)
                except:
                    tmpL.append([p,f])
                    continue
                data        = pd.DataFrame(index=tZrb,columns=feat)
                if dataM.empty:
                    if runDebug: pdb.set_trace()
                    data.loc[:,feat]     = 0
                else:
                    startM      = dataM.iloc[0,0]
                    data.loc[0:startM,feat]     = 0
                    data.loc[startM:,feat]      = zrbDict[compAcr][feat[0]]
                (comp,idx)  = watersystem.getComponent(compAcr)
                comp        = importFeatures(comp,data,pNewLab)
            elif f in dataD.keys():
                compAcr     = dataD[f]['component']
                compVL      = dataD[f]['varList']
                data        = pd.read_csv(os.path.join(pDir,p,f), sep=' ', names = compVL )
                data.replace(dataD[f]['nanValues'],np.nan,inplace=True)
                data.set_index(tZrb,inplace = True)
                for vv in ['en_twh','ed_twh']:
                    if vv in compVL:
                        data = applyCoeff(data,vv,12,'/')
                (comp,idx)  = watersystem.getComponent(compAcr)
                comp        = importVariables(comp,data,pNewLab)
        ##import Power feature for existing res
        for res in zrbDict:
            feat = 'Power'
            data = pd.DataFrame(index=tZrb,columns=[feat])
            if zrbDict[res]['Status'] == 'existing':
                data.loc[:,feat]     = zrbDict[res]['Power']
                (comp,idx)  = watersystem.getComponent(res)
                comp        = importFeatures(comp,data,pNewLab)
    return watersystem

def loadOtbData(watersystem):
    '''loop over scenarios and import data into the watersystem object - Omo-Turkana'''
    fileList    = os.listdir(gp.simDOtb)
    s           = gp.dafneParam['OTB']['scen']
    scenLabel   = gp.dafneParam['OTB']['scenLabel']
    dataD       = otbDataFolderDict(scenLabel)
    pathwLabels = gp.dafneParam['OTB']['pwLabels']
    #loop on files in pathw folder
    for f in fileList:
        print('processing file {}'.format(f))
        if f in dataD.keys():
            compAcr     = dataD[f]['component']
            varLabel    = dataD[f]['varLabel']
            data        = pd.read_csv(os.path.join(gp.simDOtb,f), sep=dataD[f]['separator'], header=dataD[f]['header'])
            ## NOTE: data from otb have pathways on the columns, no need to transpose.
            if 'timestart' in data.columns: data.drop(columns='timestart',inplace=True)
            if 'datetime' in data.columns:
                if (pd.to_datetime(data['datetime']) != tOtb).all():
                    raise Exception('datetime in the input file is different from time horizon')
                else:
                    data.drop(columns='datetime',inplace=True)
            data.columns= pathwLabels
            ##set time with leap year management
            if varLabel in ['fy','fd']:
                data.set_index(tOtbY,inplace = True)
            else:
                data.set_index(tOtb,inplace = True)
                # data = addTimeIndex(data,tOtb)## -> no more needed, WEF data are including 29 of Feb
            if varLabel in ['en_gwh','ed_gwh']:
                # convert from MWh to Gwh
                data = applyCoeff(data,None,1000,'/')
            elif varLabel == 'wa':
                #convert from mm/day to m3/sec
                mmday_to_m3sec  = gp.dafneParam['OTB'][compAcr]['Area'] * 10**6 / 86400 / 1000
                data            = applyCoeff(data,None,mmday_to_m3sec,'*')
            (comp,idxC)     = watersystem.getComponent(compAcr)
            (compVar,idxV)  = comp.getVariable(varLabel)
            compVar.data    = pd.concat([compVar.data,data],axis=1)
    print('::::::::::::::: {} configuration imported'.format(s))
    ###add cyclostationary references
    #omorateRegime
    refLabel        = 'refNat'
    refDf           = cycloRef2Df(os.path.join(gp.dafneParam['OTB']['refDir'],'omorateRegime.txt'),tOtb,refLabel,transpose=True)
    (comp,idxC)     = watersystem.getComponent('ODE')
    (compVar,idxV)  = comp.getVariable('tf')
    compVar.data    = pd.DataFrame()
    compVar.data[refLabel]      = refDf[refLabel]
    comp.variables[idxV]        = compVar
    watersystem.components[idxC]= comp
    ##
    #artificial flood pulse for recession agricolture
    refLabel        = 'refAFl'
    refDf           = cycloRef2Df(os.path.join(gp.dafneParam['OTB']['refDir'],'artificial_flood.txt'),tOtb,refLabel,transpose=True)
    (comp,idxC)     = watersystem.getComponent('REC')
    (compVar,idxV)  = comp.getVariable('tf')
    compVar.data    = pd.DataFrame()
    compVar.data[refLabel]      = refDf[refLabel]
    comp.variables[idxV]        = compVar
    watersystem.components[idxC]= comp
    ##
    #historical lake levels from satellite
    refLabel        = 'refHist'
    refDf           = pd.read_csv(os.path.join(gp.dafneParam['OTB']['refDir'],'TurkanaSatellite.txt'),sep=',',names=['time',refLabel],skiprows=1)
    refDf.set_index(pd.to_datetime(refDf['time']),inplace=True)
    (comp,idxC)     = watersystem.getComponent('TUR')
    (compVar,idxV)  = comp.getVariable('th')
    compVar.data    = pd.DataFrame()
    compVar.data[refLabel]      = refDf[refLabel]
    comp.variables[idxV]        = compVar
    watersystem.components[idxC]= comp
    ##
    #water demand for Kuraz irr. district and Private irr. scheme
    for ird in ['IR1','IR2']:
        (comp,idxC)     = watersystem.getComponent(ird)
        ##simulated water demand
        refLabel        = 'refSWd'
        (compVar,idxV)  = comp.getVariable('swd')
        if   ird == 'IR1':
            dfWd        = pd.read_csv(os.path.join(gp.simDOtb,'OTB_{}_a.F.114_SugarCane_FullIrr_WaterFluxes.txt'.format(scenLabel)),sep='\s+',header=0)
        elif ird == 'IR2':
            dfWd        = pd.read_csv(os.path.join(gp.simDOtb,'OTB_{}_a.F.186_Maize_FullIrr_WaterFluxes.txt'.format(scenLabel)),sep='\s+',header=0)

        dfWd.index      = pd.DatetimeIndex(pd.to_datetime(dfWd[['Year','Month','Day']]))
        compVar.data    = pd.DataFrame(index= tOtb)
        #conversion from mm to m3/sec
        mmday_to_m3sec  = gp.dafneParam['OTB'][ird]['Area'] * 10**6 / 86400 / 1000
        compVar.data[refLabel]  = dfWd['Irr'] * mmday_to_m3sec
        ##nominal water demand
        refLabel        = 'refNWd'
        (compVar,idxV)  = comp.getVariable('nwd')
        compVar.data    = pd.DataFrame(index= tOtb)
        compVar.data[refLabel]  = gp.dafneParam['OTB'][ird]['Nominal_WDemand']
    ##
    #compute reservoir surfaces
    for res in ['G3','KOY']:
        # if runDebug: pdb.set_trace()
        (comp,idxC)     = watersystem.getComponent(res)
        (compVar,idxV)  = comp.getVariable('ws')
        (comph,idxh)    = comp.getVariable('h')
        hS_values       = gp.dafneParam[watersystem.acronym]['{}_lev_surf_f'.format(res)]
        surface_data    = np.interp(comph.data.values,hS_values['level'] , hS_values['surface'])
        compVar.data    = pd.DataFrame(data=surface_data,index= tOtb,columns=pathwLabels)
        # comp.variables[idxV]        = compVar
        # watersystem.components[idxC]= comp
    ##
    return watersystem

def loadZrbObj(watersystem,simDir):
    '''import objectives values for ZRB case study'''
    watersystem.indicators = [
            du.Indicator('j_Hyd', ['Twh/y'],gp.freqZrb,'Energy', 'Hydropower production deficit in the ZRB system','horizon','mean','_mean','minimize'),
            du.Indicator('j_Env', ['(m3/sec)^2'],gp.freqZrb,'Environment', 'Deviation  from  natural  condition in the Zamebezi Delta','horizon','mean','_mean','minimize'),
            du.Indicator('j_Irr', ['-'],gp.freqZrb,'Food', 'Water deficit normalized with respect to irrigation demand across all the irrigation districts considered','horizon','mean','_mean','minimize'),
            du.Indicator('j_Cost',['BUSD'],gp.freqZrb,'Socio-Economic', 'Total discounted construction cost in billions of USD','horizon','sum','_sum','minimize')
            ]
    indLabels   = [i.label for i in watersystem.indicators]
    ### import directly from reference file
    objDf       = pd.read_csv(gp.dafneParam[watersystem.acronym]['objFile'],sep=' ',usecols=range(3,7),names=indLabels)
    pathwLab    = ['P' + str(n) for n in range(0,len(objDf))] #test
    objDf.index = pathwLab
    # identify groups and update labels
    newLabelDf  = identifyPathwGroup(watersystem,simDir)
    objDf['newLabel'] = newLabelDf['newLabel']
    objDf.set_index('newLabel', inplace=True)
    objDf.index.name = 'pathw'
    # load indicator into the watersystem object
    for c in objDf.columns:
        idx = indLabels.index(c)
        watersystem.indicators[idx].data = objDf.loc[:,c]
    return watersystem

def loadOtbObj(watersystem,wd):
    '''import objectives values for OTB case study'''
    dfL = []
    for p in otbDict:
        df = pd.read_csv(os.path.join(wd,otbDict[p]['fileName']),sep='\t',names=otbDict[p]['indLabels'])
        df['pathw'] = [otbDict[p]['labelPre'] + str(n) for n in range(0,len(df))]
        df.set_index('pathw',inplace=True)
        dfL.append(df)
    objDf  = pd.concat(dfL,sort=True)
    watersystem.indicators = [
            du.Indicator('j_Hyd', ['Gwh/y'],gp.freqOtb,'Energy', 'Hydropower production in the OTB system','horizon','mean','_mean','maximize'),
            du.Indicator('j_Env', ['(m3/sec)^2'],gp.freqOtb,'Environment', 'Deviation  from  natural  condition in the Omo Delta','horizon','mean','_mean','minimize'),
            du.Indicator('j_Rec', ['(m3/sec)^2'],gp.freqOtb,'Food', 'Deviation from the target flood requirement for Recession Agriculture','horizon','mean','_mean','minimize'),
            du.Indicator('j_Fish', ['MT'],gp.freqOtb,'Food', 'Maximum deficit of fish yield w.r.t average natural conditions in Lake Turkana','horizon','max','_max','minimize'),
            du.Indicator('j_Irr', ['-'],gp.freqOtb,'Food', 'Normalized squared water deficit in the two large scale irrigation districts','horizon','mean','_mean','minimize')
            ]
    for c in objDf.columns:
        indLabel = [i.label for i in watersystem.indicators]
        idx = indLabel.index(c)
        watersystem.indicators[idx].data = objDf.loc[:,c]
    return watersystem

###################################
#process
###################################

if __name__ == '__main__':
    ##ZRB
    zrb = loadZrbData(zrb)
    zrb = loadZrbObj(zrb,gp.simDZrb)
    save_object(zrb, os.path.join(gp.wdZrb,'pyObjects','zrb.pkl'))
    ##OTB
    otb = loadOtbData(otb)
    otb = loadOtbObj(otb,gp.simDOtb)
    save_object(otb, os.path.join(gp.dafneParam['OTB']['pklDir'],'otb.pkl'))
    print('import from WEF completed!')
