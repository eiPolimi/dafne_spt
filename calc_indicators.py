"""
DAFNE Project - Simulation Processing Toolbox
---------------------------------------------
:title:     Calc Indicators
:authors:   MarcoMicotti@polimi
:content:   Indicators computation for Food, Energy and Water-Ecosystem sectors

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
import sim_importer
import globalparam as gp
import importlib
importlib.reload(du)
importlib.reload(il)
importlib.reload(sim_importer)

################################################################################
# init & load data and parameters
################################################################################
wd          = gp.wd
runDebug    = True
# runDebug    = False

caseStudy   = { 'ZRB'    : False,
                'OTB'    : True }

# runType     = 'updateZrb'
# runType     = 'initZrb'
runType     = 'updateOtb'
# runType     = 'initOtb'

######################################
#util functions
######################################

def sumCompTs(watersystem, pathwFilter, label,type='indicator'):
    dfL = []
    for c in watersystem.components:
        if type == 'indicator':
            ts,idx = c.getIndicator(label)
        elif type == 'feature':
            ts,idx = c.getFeature(label)
        if ts:
            df = ts.data.reindex(columns=pathwFilter)
            # df.columns = df.columns + '_' + c.acronym
            dfL.append(df)
    tsDf        = pd.concat(dfL,axis=1,sort=False)
    patDf       = tsDf.T.groupby(tsDf.columns).sum().T
    return patDf

def plotInd(df,nRef,saveDir,figTitle,colorsD={},blLabel=''):
    '''plot dataframe, with reference, custom color scale. Not sure about blLabel...'''
    if colorsD:
        ax = df.iloc[:,nRef:].plot(figsize=(16,9),c=colorsD['all'],legend=False)
        ax = df.iloc[:,0:nRef].plot(ls='--',c=colorsD['ref'],ax=ax)
        if 'mean' in df.columns:
            ax = df.loc[:,'mean'].plot(c=colorsD['ref_mean'],linewidth=3,ax=ax,legend=True)
            # if runDebug: pdb.set_trace()
    else:
        ax = df.iloc[:,nRef:].plot(figsize=(16,9),legend=False)
        ax = df.iloc[:,0:nRef].plot(ls='--',c='#000000',ax=ax)
    if blLabel:
        ax = df.loc[:,blLabel].plot(c='#e32267',linewidth=4,legend=True,ax=ax)
    ax.set_title(figTitle)
    # plt.show(block=False)
    plt.savefig(os.path.join(saveDir,figTitle.replace(' ','') + '.png'))
    plt.close()

def plotLoopPathw(df,nRef,imgDir,vrb,comp,plotRange=True):
    '''Loop over the columns, to plot single pathways behavior against reference and, optionally, together with ranges'''
    if plotRange:
        vrbmax,idx     = comp.getIndicator(vrb.label.replace('Mean','Max'))
        vrbmin,idx     = comp.getIndicator(vrb.label.replace('Mean','Min'))
    for p in df.columns[nRef:]:
        if vrb.type == 'Indicator':
            descr   = vrb.descr
        else:
            descr   = vrb.type
        figTitle    = '{} - {} - Pathway {}'.format(descr,comp.name,p)
        fileName    = '{}_{}_{}'.format(vrb.label,comp.acronym,p)
        fig, ax     = plt.subplots(figsize=(8,6))
        ax = df.loc[:,p].plot()
        ax = df.iloc[:,0:nRef].plot(ls='--',c='#000000',ax=ax)
        if plotRange:
            ax.fill_between(df.index, vrbmin.data[p], vrbmax.data[p], alpha=0.2)
        ax.set_title(figTitle)
        ax.set_ylabel(vrb.unit)
        plt.savefig(os.path.join(imgDir,fileName.replace('.','') + '.png'))
        plt.close()
    return print('{} - {}: loop on Pathways done'.format(comp.name,vrb.label))


def export365ts(watersystem,pathwFilter):
    '''export and save 365 plots of timeseries'''
    varList = ['h','r','tr','en_twh','ed_twh']
    compList= [c for c in watersystem.components if 'Reservoir' in c.type]
    colorD  = {'all':'#bab9b9','ref': '#000000','ref_mean':'#058b9d'}
    for vl in varList:
        for c in compList:
            v,idxV     = c.getVariable(vl)
            df         = v.data.reindex(columns=pathwFilter)
            for p in df.columns:
                df['moy']   = df.index.month
                df['year']  = df.index.year
                dfP         = df.pivot(index='moy',columns='year',values=p)
                dfP.insert(0,'mean',dfP.mean(axis=1))
                nRef = 1
                if ('Reservoir' in c.type):
                    if vl == 'h':
                        dfP.insert(0,'max_ol',gp.refLevels[c.acronym]['max_ol'])
                        dfP.insert(0,'min_ol',gp.refLevels[c.acronym]['min_ol'])
                        nRef += 2
                yearStart   = gp.dafneParam[watersystem.acronym]['tStart'].year
                yearEnd     = gp.dafneParam[watersystem.acronym]['tEnd'].year
                figTitle    = '{} - {} {} {}'.format(p,c.acronym,v.frequency, v.type)
                #save csv
                pDir        = os.path.join(gp.dafneParam[watersystem.acronym]['dashDir'],p)
                os.makedirs(pDir,exist_ok=True)
                dfP.to_csv(os.path.join(pDir,figTitle.replace(p+'-','').replace(' ','') + '.csv'))
                # save png
                pImgDir     = os.path.join(pDir,'img')
                os.makedirs(pImgDir,exist_ok=True)
                plotInd(dfP,nRef,pImgDir,figTitle,colorD,blLabel='')
    return
    ##

######################################
#sector functions
######################################
def compEnergyIndicators(comp):
    variableList    = [v.label for v in comp.variables]
    #res water level cyclo indicators
    try:
        idxH            = variableList.index('h')
        dfH             = comp.variables[idxH].data
        idxS            = variableList.index('s')
        dfS             = comp.variables[idxS].data
        if dfH.empty:
            print('warning: {} no water level variable'.format(comp.name))
            return comp
        if comp.system == 'ZRB':
            comp.indicators.append(il.i_en_wlcyclo_m_mean(dfH.copy(),variableList[idxH]))
            comp.indicators.append(il.i_en_wlcyclo_m_max(dfH.copy(),variableList[idxH]))
            comp.indicators.append(il.i_en_wlcyclo_m_min(dfH.copy(),variableList[idxH]))
            comp.indicators.append(il.i_en_scyclo_m_mean(dfS.copy()))
        elif comp.system == 'OTB':
            comp.indicators.append(il.i_en_wlcyclo_mean(dfH.copy(),variableList[idxH]))
            comp.indicators.append(il.i_en_wlcyclo_min(dfH.copy(),variableList[idxH]))
            comp.indicators.append(il.i_en_wlcyclo_max(dfH.copy(),variableList[idxH]))
    except :
        pass
    #res release cyclo indicators
    try:
        idxR            = variableList.index('r')
        dfR             = comp.variables[idxR].data
        if dfR.empty:
            return comp
        if comp.system == 'ZRB':
            comp.indicators.append(il.i_en_rcyclo_m_mean(dfR.copy()))
        elif comp.system == 'OTB':
            comp.indicators.append(il.i_en_rcyclo_mean(dfR.copy()))
    except :
        pass
    #production indicators
    if comp.system == 'ZRB': idx    = variableList.index('en_twh')
    if comp.system == 'OTB': idx    = variableList.index('en_gwh')
    dfEn            = comp.variables[idx].data
    uom             = comp.variables[idx].unit
    if dfEn.empty:
        if runDebug: pdb.set_trace()
        print('{} not enough variables found for energy deficit indicators'.format(comp.name))
        return comp
    comp.indicators.append(il.i_en_production_m_mean(dfEn.copy(),uom))
    comp.indicators.append(il.i_en_production_y_sum(dfEn.copy(),uom))
    comp.indicators.append(il.i_en_production_sum(dfEn.copy(),uom))
    # comp.indicators.append(il.i_en_production_d_mean(dfEn.copy(),uom))
    comp.indicators.append(il.i_en_production_y_mean(dfEn.copy(),uom))
    #start of operation indicators
    if comp.system == 'ZRB':
        comp.indicators.append(il.i_en_firstoperatingyear(dfEn.copy()))
        comp.indicators.append(il.i_en_firstoperatingday(dfEn.copy(),comp.acronym))
    #deficit indicators
    # get energy demand
    if comp.system == 'ZRB': idx    = variableList.index('ed_twh')
    if comp.system == 'OTB': idx    = variableList.index('ed_gwh')
    dfEnD           = comp.variables[idx].data
    if dfEnD.empty:
        return comp
    comp.indicators.append(il.i_en_demand_y_sum(dfEnD.copy(),uom))
    dfEnDef  = dfEnD - dfEn
    for c in dfEnDef.columns: dfEnDef[c]   = dfEnDef[c].apply(lambda x: max(x,0 ))
    comp.indicators.append(il.i_en_deficit_m_mean(dfEnDef.copy(),uom))
    comp.indicators.append(il.i_en_deficit_y_sum(dfEnDef.copy(),uom))
    comp.indicators.append(il.i_en_deficit_y_mean(dfEnDef.copy(),uom))
    comp.indicators.append(il.i_en_deficit_sum(dfEnDef.copy(),uom))
    return comp

def compFoodIndicators(comp):
    variableList    = [v.label for v in comp.variables]
    idxwa           = variableList.index('wa')
    dfWa            = comp.variables[idxwa].data

    if comp.system == 'OTB':
        idxnwd          = variableList.index('nwd')
        idxswd          = variableList.index('swd')
        dfSWd           = comp.variables[idxswd].data
        dfNWd           = comp.variables[idxnwd].data
        if dfWa.empty | dfNWd.empty | dfSWd.empty:
            print('{} not enough variables found for irrigation indicators'.format(comp.name))
            return comp
        if dfNWd.shape[1] == 1:
            dfNWd = pd.DataFrame(data = np.tile(dfNWd,len(dfWa.columns)),index=dfNWd.index,columns = dfWa.columns)
        if dfSWd.shape[1] == 1:
            dfSWd = pd.DataFrame(data = np.tile(dfSWd,len(dfWa.columns)),index=dfSWd.index,columns = dfWa.columns)
        #cyclostationary ind, only for OTB
        dfWaMovAv = dfWa.rolling(5,1,center=True).mean()
        comp.indicators.append(il.i_irr_qcyclo_mean(dfWaMovAv.copy(),variableList[idxwa]))
        comp.indicators.append(il.i_irr_qcyclo_max(dfWaMovAv.copy(),variableList[idxwa]))
        comp.indicators.append(il.i_irr_qcyclo_min(dfWaMovAv.copy(),variableList[idxwa]))
        comp.indicators.append(il.i_irr_qcyclo_mean(dfSWd.copy(),variableList[idxswd]))
        #volume indicators
        comp.indicators.append(il.i_irr_abstraction_m_mean(dfWa.copy(),comp.variables[idxwa].label))
        comp.indicators.append(il.i_irr_abstraction_m_mean(dfSWd.copy(),comp.variables[idxswd].label))
        comp.indicators.append(il.i_irr_abstraction_y_sum(dfWa.copy(),gp.dafneParam[comp.system]['freq']))
        comp.indicators.append(il.i_irr_wdemand_y_sum(dfSWd.copy(),gp.dafneParam[comp.system]['freq']))
        # deficit indicators wrt to simulated water demand
        dfSWDef          = dfSWd - dfWa
        for c in dfSWDef.columns: dfSWDef[c]   = dfSWDef[c].apply(lambda x: max(x,0 ))
        comp.indicators.append(il.i_irr_deficit_m_mean(dfSWDef.copy()))
        comp.indicators.append(il.i_irr_deficit_y_sum(dfSWDef.copy()))
        # comp.indicators.append(il.i_irr_deficit_sum(dfSWDef.copy()))
        comp.indicators.append(il.i_irr_deficit_y_mean(dfSWDef.copy(),gp.dafneParam[comp.system]['freq']))
    elif comp.system == 'ZRB':
        idxnwd          = variableList.index('wd')
        dfNWd           = comp.variables[idxnwd].data
        if dfWa.empty | dfNWd.empty:
            print('{} not enough variables found for irrigation indicators'.format(comp.name))
            return comp
        # #cyclostationary indicators
        #volume indicators
        comp.indicators.append(il.i_irr_abstraction_m_mean(dfWa.copy(),comp.variables[idxwa].label))
        comp.indicators.append(il.i_irr_abstraction_m_mean(dfNWd.copy(),comp.variables[idxnwd].label))
        comp.indicators.append(il.i_irr_abstraction_y_sum(dfWa.copy(),gp.dafneParam[comp.system]['freq']))
        comp.indicators.append(il.i_irr_abstraction_y_mean(dfWa.copy(),gp.dafneParam[comp.system]['freq']))
        comp.indicators.append(il.i_irr_wdemand_y_sum(dfNWd.copy(),gp.dafneParam[comp.system]['freq']))
        #deficit indicators wrt to nominal water demand
        dfNWDef          = dfNWd - dfWa
        for c in dfNWDef.columns: dfNWDef[c]   = dfNWDef[c].apply(lambda x: max(x,0 ))
        comp.indicators.append(il.i_irr_deficit_m_mean(dfNWDef.copy()))
        comp.indicators.append(il.i_irr_deficit_y_sum(dfNWDef.copy()))
        # comp.indicators.append(il.i_irr_deficit_sum(dfNWDef.copy()))
        comp.indicators.append(il.i_irr_deficit_y_mean(dfNWDef.copy(),gp.dafneParam[comp.system]['freq']))
    #
    return comp

    #deficit indicator for recession
def compRecessionIndicators(comp):
    variableList    = [v.label for v in comp.variables]
    idxq            = variableList.index('q')
    idxtf           = variableList.index('tf')
    dfQ             = comp.variables[idxq].data
    dfTf            = comp.variables[idxtf].data
    if dfQ.empty or dfTf.empty:
        print('{} not enough variables found for recession indicators'.format(comp.name))
        return comp
    dfQMovAv = dfQ.rolling(5,1,center=True).mean()
    comp.indicators.append(il.i_irr_qcyclo_mean(dfQMovAv.copy(),variableList[idxq]))
    comp.indicators.append(il.i_irr_qcyclo_max(dfQMovAv.copy(),variableList[idxq]))
    comp.indicators.append(il.i_irr_qcyclo_min(dfQMovAv.copy(),variableList[idxq]))
    comp.indicators.append(il.i_irr_qcyclo_mean(dfTf.copy(),variableList[idxtf]))
    dfQDef          = dfTf.values - dfQ
    #manage period
    startDoy        = dt.datetime(1900,8,29).timetuple().tm_yday
    endDoy          = dt.datetime(1900,9,15).timetuple().tm_yday
    idx             = (dfQDef.index.dayofyear >= startDoy ) & (dfQDef.index.dayofyear <= endDoy )
    dfQDef.loc[~idx,:] = np.nan
    #positive distance
    dfQDefPos               = dfQDef.copy()
    dfQDefPos[dfQDefPos<0]  = np.nan
    comp.indicators.append(il.i_irr_distancepos_y_d_mean(dfQDefPos.copy()))
    comp.indicators.append(il.i_irr_distancepos_mean(dfQDefPos.copy()))
    #negative distance
    dfQDefNeg               = dfQDef.copy()
    dfQDefNeg[dfQDefNeg>0]  = np.nan
    dfQDefNeg               = dfQDefNeg.abs()
    comp.indicators.append(il.i_irr_distanceneg_y_d_mean(dfQDefNeg.copy()))
    comp.indicators.append(il.i_irr_distanceneg_mean(dfQDefNeg.copy()))
    return comp
#
def compWaterIndicators(comp):
    variableList    = [v.label for v in comp.variables]
    idxq            = variableList.index('q')
    idxtf           = variableList.index('tf')
    dfQ             = comp.variables[idxq].data
    dfTf            = comp.variables[idxtf].data
    if dfQ.empty or dfTf.empty:
        # import pdb; pdb.set_trace()
        print('{} not enough variables found for environmental indicators'.format(comp.name))
        return comp
    if comp.system == 'OTB':
        #cyclostationary ind
        dfQMovAv = dfQ.rolling(5,1,center=True).mean()
        comp.indicators.append(il.i_env_qcyclo_mean(dfQMovAv.copy(),variableList[idxq]))
        comp.indicators.append(il.i_env_qcyclo_max(dfQMovAv.copy(),variableList[idxq]))
        comp.indicators.append(il.i_env_qcyclo_min(dfQMovAv.copy(),variableList[idxq]))
        comp.indicators.append(il.i_env_qcyclo_mean(dfTf.copy(),variableList[idxtf]))
        #distance
        dfQDef          = dfTf.values - dfQ
    elif comp.system == 'ZRB':
        #cyclostationary ind
        comp.indicators.append(il.i_env_qcyclo_m_mean(dfQ.copy(),variableList[idxq]))
        comp.indicators.append(il.i_env_qcyclo_m_mean(dfTf.copy(),variableList[idxtf]))
        #distance
        dfQDef          = dfTf - dfQ
        #manage periodicity
        startDoy        = dt.datetime(1900,2,1).timetuple().tm_yday
        endDoy          = dt.datetime(1900,3,31).timetuple().tm_yday
        idx             = (dfQDef.index.dayofyear >= startDoy ) & (dfQDef.index.dayofyear <= endDoy )
        dfQDef.loc[~idx,:] = np.nan
    #deficit indicators -> disabled because duplicates wrt IHA
    #positive distance
    dfQDefPos               = dfQDef.copy()
    dfQDefPos[dfQDefPos<0]  = np.nan
    comp.indicators.append(il.i_env_distancepos_m_mean(dfQDefPos.copy()))
    comp.indicators.append(il.i_env_distancepos_y_mean(dfQDefPos.copy()))
    comp.indicators.append(il.i_env_distancepos_mean(dfQDefPos.copy()))
    #negative distance
    # dfQDefNeg               = dfQDef.copy()
    # dfQDefNeg[dfQDefNeg>0]  = np.nan
    # dfQDefNeg               = dfQDefNeg.abs()
    # comp.indicators.append(il.i_env_distanceneg_m_mean(dfQDefNeg.copy()))
    # comp.indicators.append(il.i_env_distanceneg_y_mean(dfQDefNeg.copy()))
    # comp.indicators.append(il.i_env_distanceneg_mean(dfQDefNeg.copy()))
    return comp

def compLakeIndicators(comp):
    variableList    = [v.label for v in comp.variables]
    #deficit indicators
    idxH            = variableList.index('h')
    dfH             = comp.variables[idxH].data
    if dfH.empty:
        print('{} not enough variables found for lake indicators'.format(comp.name))
    else:
        #cyclostationary ind
        dfHMovAv = dfH.rolling(5,1,center=True).mean()
        comp.indicators.append(il.i_env_wlcyclo_mean(dfHMovAv.copy()))
    #fish production indicators
    idxF            = variableList.index('fy')
    dfF             = comp.variables[idxF].data
    comp.indicators.append(il.i_irr_flake_y(dfF.copy()))
    comp.indicators.append(il.i_irr_flake_sum(dfF.copy()))
    #fish deficit indicators
    idxFd           = variableList.index('fd')
    dfFd            = comp.variables[idxFd].data
    comp.indicators.append(il.i_irr_fdeficit_y(dfFd.copy()))
    comp.indicators.append(il.i_irr_fdeficit_sum(dfFd.copy()))
    return comp

def compFishInReservoirIndicators(comp,ws_acr):
    variableList    = [v.label for v in comp.variables]
    #deficit indicators
    idxWs           = variableList.index('ws')
    dfWs            = comp.variables[idxWs].data
    if dfWs.empty:
        print('{} not enough data for fish in reservoir indicators'.format(comp.name))
        return comp
    #compute yearly average area and yearly fish production
    coeff           = gp.dafneParam[ws_acr]['FishInRes']['coeff']
    exp             = gp.dafneParam[ws_acr]['FishInRes']['exp']
    dfWs_y_mean     = il.periodicStat(dfWs,'yearly','mean')
    dfFRes_y_mean   = coeff * (dfWs_y_mean ** exp )
    #fish production indicators
    comp.indicators.append(il.i_irr_fishres_y(dfFRes_y_mean.copy()))
    comp.indicators.append(il.i_irr_fishres_mean(dfFRes_y_mean.copy()))
    comp.indicators.append(il.i_irr_fishres_sum(dfFRes_y_mean.copy()))
    return comp

def computeIndicators(watersystem):
    for i,c in enumerate(watersystem.components):
        print('Computing indicator for {} component ...'.format(c.name))
        if 'Hydro Power' in c.type:
            if 'Victoria' in c.name:
                continue
            else:
                watersystem.components[i] = compEnergyIndicators(c)
            if c.acronym in ['KOY','G3']:
                # if runDebug: pdb.set_trace()
                watersystem.components[i] = compFishInReservoirIndicators(c,watersystem.acronym)
        elif 'Recession' in c.name:
            watersystem.components[i] = compRecessionIndicators(c)
        elif 'Irrigation' in c.name:
            # pass
            watersystem.components[i] = compFoodIndicators(c)
        elif 'Ecosystem' in c.type:
            watersystem.components[i] = compWaterIndicators(c)
        elif 'lake' in c.type:
            if runDebug: pdb.set_trace()
            watersystem.components[i] = compLakeIndicators(c)
        print('..... done!')
    return watersystem

def exportZrb2Dashboard(watersystem,pathwFilter):
    #energy
    enProdDf = sumCompTs(watersystem, pathwFilter, 'i_E_2_EnProd_SumY')
    enDDf    = sumCompTs(watersystem, pathwFilter, 'i_E_EnDemand_SumY')
    enProdDf.index.rename('years',inplace=True)
    enProdDf.insert(0,'Energy Demand',enDDf.iloc[:,0])
    if plotDash:
        plotInd(enProdDf,1,'Yearly sum of Energy Production in the ZRB')
    if runDebug: pdb.set_trace()
    enProdDf.to_csv(os.path.join(gp.wdZrb,gp.csvDashDir,'ZRB_EnProduction.csv'))
    #irrigation deficit
    irrWaDf  = sumCompTs(watersystem, pathwFilter, 'i_F_IrrWa_SumY')
    irrWdDf  = sumCompTs(watersystem, pathwFilter, 'i_F_IrrWd_SumY')
    irrWaDf.index.rename('years',inplace=True)
    irrWaDf.insert(0,'Water Demand',irrWdDf.iloc[:,0])
    if plotDash:
        print('No plot: check better plotInd function ')
        # plotInd(irrWaDf,1,'Yearly sum of Water abstraction in the ZRB')
    irrWaDf.to_csv(os.path.join(gp.wdZrb,gp.csvDashDir,'ZRB_IrrWAbstraction.csv'))
    #water abstraction cyclo mean
    irrWaDf  = sumCompTs(watersystem, pathwFilter, 'i_F_IrrMeanM_wa')
    irrWdDf  = sumCompTs(watersystem, pathwFilter, 'i_F_IrrMeanM_wd')
    irrWaDf.index.rename('months',inplace=True)
    irrWaDf.insert(0,'Water Demand',irrWdDf.iloc[:,0])
    irrWaDf.to_csv(os.path.join(gp.wdZrb,gp.csvDashDir,'ZRB_IrrWAbstractionMeanM.csv'))
    #water abstraction single  pathway
    irrWaDfSingleP = pd.DataFrame(columns=pathwFilter)
    for id in [c for c in watersystem.components if c.acronym.startswith('ID')]:
        i,idxI    = id.getIndicator('i_F_IrrWa_MeanY')
        df        = i.data.reindex(columns=pathwFilter)
        irrWaDfSingleP.loc[id.acronym,:] = df.values
    for p in irrWaDfSingleP.columns:
        ax = irrWaDfSingleP.loc[:,p].plot.bar(rot=0)
        ax.set_title('Yearly average volume of water supply for irrigation')
        ax.set_ylabel('Mm3')
        ax.set_xlabel('Equivalent Irrigation District')
        # plt.show(block=False)
        # if runDebug: pdb.set_trace()
        plt.savefig(os.path.join(gp.dafneParam[watersystem.acronym]['imgDir'],'{}_IrrWaY.png'.format(p)))
        plt.close()
    #installed power
    # powDf = sumCompTs(watersystem, pathwFilter, 'Power', type='feature')
    # powDf.index.rename('years',inplace=True)
    # powDf.to_csv(os.path.join(gp.wdZrb,gp.csvDashDir,'ZRB_InstalledPower.csv'))
    #Zambezi Delta
    c,idx       = watersystem.getComponent('ZDE')
    q,idxq      = c.getIndicator('i_W_QCycloMMean_q')
    tf,idxtf    = c.getIndicator('i_W_QCycloMMean_tf')
    dfRef       = pd.DataFrame(tf.data.iloc[:,0])
    dfRef.columns = ['RefNat']
    dfOut       = pd.concat([dfRef,q.data.reindex(columns=pathwFilter)],axis=1)
    dfOut.index.name = 'months'
    # if plotDash:
    #     plotInd(dfOut,1,'Monthly average of streamflow in the Zambezi Delta')
    dfOut.to_csv(os.path.join(gp.wdZrb,gp.csvDashDir,c.acronym + '_QCycloMMean.csv'))
    #####timing
    dfTL = {}
    for r in [c for c in watersystem.components if c.acronym in ['BAT','DEV','MNK']]:
        i,idxI    = r.getIndicator('i_E_day1EnProd')
        df        = i.data.reindex(columns=pathwFilter)
        dfTL[r.acronym] = df
    dfOut = pd.concat(dfTL).T
    dfOut.columns = dfOut.columns.levels[0]
    dfOut.insert(0,'pathw',dfOut.index)
    dfOut.set_index(np.array(range(1,len(dfOut)+1)),drop=False,inplace = True)
    dfOut.index.names = ['pathw_n']
    dfOut.to_csv(os.path.join(gp.dafneParam[watersystem.acronym]['dashDir'],watersystem.acronym + '_' + 'timing.csv'))
    #####res cyclostationary values
    resList   = [c for c in watersystem.components if 'Reservoir' in c.type]
    enProdDf  = pd.DataFrame(columns=pathwFilter)
    for r in resList:
        for indLab in ['i_E_HCycloMeanM_h','i_E_2_EnProd_SumY', 'i_E_SCycloMeanM','i_E_2_EnProd_MeanY']:
            i,idxI    = r.getIndicator(indLab)
            df        = i.data.reindex(columns=pathwFilter)
            if indLab == 'i_E_HCycloMeanM_h':
                df.insert(0,'max_ol',gp.refLevels[r.acronym]['max_ol'])
                if r.acronym in ['DEV','MNK']:
                    nRef = 1
                else:
                    df.insert(0,'min_ol',gp.refLevels[r.acronym]['min_ol'])
                    nRef = 2
                df.index.name = 'month'
                # if runDebug: pdb.set_trace()
                plotLoopPathw(df,nRef,gp.dafneParam[watersystem.acronym]['imgDir'],i,r)
            elif indLab == 'i_E_2_EnProd_MeanY':
                enProdDf.loc[r.acronym,:] = df.values
            else:
                df.index.name = 'month'
                nRef = 0
            df.to_csv(os.path.join(gp.dafneParam[watersystem.acronym]['dashDir'],r.acronym + '_' + indLab + '.csv'))
    for p in enProdDf.columns:
        ax = enProdDf.loc[:,p].plot.bar(rot=0)
        ax.set_title('Yearly average energy production')
        ax.set_ylabel('TWh/y')
        ax.set_xlabel('Power Plants')
        # plt.show(block=False)
        plt.savefig(os.path.join(gp.dafneParam[watersystem.acronym]['imgDir'],'{}_EneProductionY.png'.format(p)))
        plt.close()
    ##
    return print('export to dashboard completed')

def exportOtb2Dashboard(watersystem,pathwFilter):
    imgDir      = gp.dafneParam[watersystem.acronym]['imgDir']
    dfEnL       = []
    dfFresL     = []
    for c in watersystem.components:
        if c.acronym == 'ODE':
            q,idxq      = c.getIndicator('i_W_QCycloMean_q')
            tf,idxtf    = c.getIndicator('i_W_QCycloMean_tf')
            #add recession
            rectf,idxtf = watersystem.getComponent('REC')[0].getIndicator('i_F_QCycloMean_tf')
            dfOut       = pd.concat([tf.data,rectf.data,q.data.reindex(columns=pathwFilter)],axis=1)
            dfOut.index.name = 'doy'
            plotLoopPathw(dfOut,2,imgDir,q,c)
            dfOut.round(2).to_csv(os.path.join(gp.dafneParam['OTB']['dashDir'],c.acronym + '_QCycloMean.csv'))
        elif c.acronym == 'REC':
            q,idxq      = c.getIndicator('i_F_QCycloMean_q')
            tf,idxtf    = c.getIndicator('i_F_QCycloMean_tf')
            dfOut       = pd.concat([tf.data,q.data.reindex(columns=pathwFilter)],axis=1)
            dfOut.index.name = 'doy'
            plotLoopPathw(dfOut,1,imgDir,q,c)
            dfOut.round(2).to_csv(os.path.join(gp.dafneParam['OTB']['dashDir'],c.acronym + '_QCycloMean.csv'))
        elif c.acronym == 'TUR':
            #lake levels
            h,idxh      = c.getVariable('h')
            th,idxth    = c.getVariable('th')
            dfOut       = pd.concat([th.data,h.data.reindex(columns=pathwFilter)],axis=1)
            dfOut.index.name = 'time'
            plotLoopPathw(dfOut,1,imgDir,h,c,plotRange=False)
            dfOut.round(2).to_csv(os.path.join(gp.dafneParam['OTB']['dashDir'],c.acronym + '_levels.csv'))
            #fish yield
            fy,idxfy    = c.getVariable('fy')
            dfOut       = fy.data.reindex(columns=pathwFilter)
            dfOut.insert(0,'refNat',gp.dafneParam[watersystem.acronym]['TurFishNat'])
            plotLoopPathw(dfOut,1,imgDir,fy,c,plotRange=False)
            dfOut.round(2).to_csv(os.path.join(gp.dafneParam['OTB']['dashDir'],c.acronym + '_fishyield.csv'))
        elif c.acronym in ['IR1','IR2']:
            wa,idxwa      = c.getIndicator('i_F_QCycloMean_wa')
            nwd,idxwd     = c.getVariable('nwd')
            swd,idxwd     = c.getVariable('swd')
            nwd_doy       = il.periodicStat(nwd.data,'daily','mean')
            swd_doy       = il.periodicStat(swd.data,'daily','mean')
            dfOut         = pd.concat([nwd_doy,swd_doy,wa.data.reindex(columns=pathwFilter)],axis=1)
            dfOut.index.name = 'doy'
            plotLoopPathw(dfOut,2,imgDir,wa,c)
            dfOut.round(2).to_csv(os.path.join(gp.dafneParam['OTB']['dashDir'],c.acronym + '_QCycloMean.csv'))
        elif c.acronym in ['G3','KOY']:
            #water level
            h,idxh      = c.getIndicator('i_E_HCycloMean_h')
            dfOut       = h.data.reindex(columns=pathwFilter)
            dfOut.insert(0,'max_ol',gp.refLevels[c.acronym]['max_ol'])
            dfOut.insert(0,'min_ol',gp.refLevels[c.acronym]['min_ol'])
            dfOut.index.name = 'doy'
            plotLoopPathw(dfOut,2,imgDir,h,c)
            dfOut.round(2).to_csv(os.path.join(gp.dafneParam['OTB']['dashDir'],c.acronym + '_levels.csv'))
            #energy
            en,idxen    = c.getIndicator('i_E_2_EnProd_MeanY')
            dfOut       = en.data.reindex(columns=pathwFilter)
            dfOut.index = [c.name]
            dfEnL.append(dfOut)
            #fish in res
            fres,idxen  = c.getIndicator('i_F_IrrFishRes_Mean')
            dfOut       = fres.data.reindex(columns=pathwFilter)
            dfOut.index = [c.name]
            dfFresL.append(dfOut)
    if len(dfEnL) > 0 :
        dfEn = pd.concat(dfEnL,axis=0)
        if runDebug: pdb.set_trace()
        dfEn.round(2).to_csv(os.path.join(gp.dafneParam['OTB']['dashDir'],'EnProduction_Y_mean.csv'))
    if len(dfFresL) > 0 :
        dfFres = pd.concat(dfFresL,axis=0)
        dfFres.round(2).to_csv(os.path.join(gp.dafneParam['OTB']['dashDir'],'FishRes_Y_mean.csv'))
    return print('export to dashboard completed')

def exportWs2Geop(watersystem,pathwFilter,varFilter=[],indFilter=[]):
    for c in watersystem.components:
        print(c.name)
        c.export2Geop(pathwFilter,varFilter,indFilter)
    print('export of the {} water system to csv files completed!!'.format(watersystem.name))

def export2MPVT(watersystem,type,pathwFilter=[]):
    '''export objectives to screening tool or horizon indicators to MPVT tool'''
    if type     == 'objectives':
        iList       = watersystem.indicators
        fileName    = 'allObjGlNormFlip'
        # funDirL     = ['minimize']*len(iList)
    elif type   == 'horizon_indicators':
        iList       = []
        iLabList    = []
        iDescrList  = []
        iSubBasinL  = []
        iCountryL   = []
        for c in watersystem.components:
            compHorizonInd = [i for i in c.indicators if i.frequency == 'horizon']
            iList       += compHorizonInd
            iLabList    += ['{}_{}'.format(i.label,c.acronym) for i in compHorizonInd]
            iDescrList  += ['{} {}'.format(i.descr,c.name) for i in compHorizonInd]
            iSubBasinL  += [c.subbasin for i in range(0,len(compHorizonInd))]
            iCountryL   += [c.country for i in range(0,len(compHorizonInd))]
        fileName    = 'allIndicatorsNormFlip'
    metaLabels  = ['name', 'short_description','long_description','subbasin','country','unit','origin','sector','scenario','url','funDirection','max','min']
    pDict       = gp.dafneParam[watersystem.acronym]
    metaInfo    = { 'baseurl'  : pDict['indUrl'],
                    'origin'   : gp.dafneParam[watersystem.acronym]['origin'],
                    'scenario' : pDict['scenLabel']
                }
    saveDir     = pDict['pcDir']
    dfL     = []
    dfMetaL = []
    funDirL = []
    for k,i in enumerate(iList): #All the b
        url_suffix  = ''.join(i.label.split('_')[0:-1]).lower()
        i_url       = metaInfo['baseurl'] + url_suffix
        # plot value function, to be checked
        # plotminmaxVF(i, gp.dafneParam[watersystem.acronym]['imgDir'],False)
        if type   == 'objectives':
            dfL.append(i.data)
            dfMetaL.append(pd.DataFrame([i.label, i.descr, i.unit, metaInfo['origin'], i.sector, metaInfo['scenario'], i_url,i.funDirection]))
        elif type == 'horizon_indicators':
            if ('year1EnProd' in iLabList[k]) | ('day1EnProd' in iLabList[k]): continue
            print(iLabList[k])
            if iLabList[k] == 'i_F_IrrFishProd_Sum_TUR': import pdb; pdb.set_trace()
            funDirL.append(i.funDirection)
            i.data.index = [iLabList[k]]
            if pathwFilter == []:
                dfL.append(i.data.T)
            else:
                dfL.append(i.data[pathwFilter].T)
            imax = round(dfL[-1].max().values[0],2)
            imin = round(dfL[-1].min().values[0],2)
            dfMetaL.append(pd.DataFrame([iLabList[k], '', iDescrList[k],iSubBasinL[k],iCountryL[k], i.unit, metaInfo['origin'], i.sector, metaInfo['scenario'], i_url,i.funDirection, imax,imin]))
    dfMetaOut           = pd.concat(dfMetaL,axis = 1,sort = False)
    dfMetaOut.index     = metaLabels
    dfMetaOut.columns   = dfMetaOut.iloc[0,:]
    dfOut               = pd.concat(dfL,axis = 1,sort = False).T
    if (watersystem.acronym == 'OTB') & (type == 'objectives'):
        #workaround to remove 10^6 factor in the j_Env
        dfOut.loc['j_Env',:] = dfOut.loc['j_Env',:].apply(lambda x: x*(10**6))
        #workaround to remove 10^3 factor in the j_Rec
        dfOut.loc['j_Rec',:] = dfOut.loc['j_Rec',:].apply(lambda x: x*(10**3))
        #normalize
        dfOutN    = du.minmaxNorm(dfOut,funDirL,append=True,flip=True)
        #workaround to get the production instead of opposite produtction
        dfOutN.loc['j_Hyd',:] = dfOutN.loc['j_Hyd',:].abs()
    else: #
        if runDebug: pdb.set_trace()
        dfOutN    = du.minmaxNorm(dfOut,funDirL,append=True,flip=True)
    #valuefunction plot
    # plotValueFunctio(dfOut,funDirL,saveDir)
    dfExport  = pd.concat([dfMetaOut,dfOutN.T.astype(float).round(2)],axis=0)
    outputDir = os.path.join(saveDir,metaInfo['scenario'])
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    dfExport.to_csv(os.path.join(outputDir,watersystem.acronym + '_' + fileName + '.csv'), header=False)
    print('export of {} {} completed!!'.format(watersystem.acronym,type))

######################################
#process
######################################
pFilterZrb     = gp.dafneParam['ZRB']['pwLabels'] #mapping of 2019 NSL selection with new labels
gp.dafneParam['OTB']['pwLabels'].sort()
pFilterOtb     = gp.dafneParam['OTB']['pwLabels']

if __name__ == '__main__':
    if caseStudy['ZRB']:
        if runType == 'initZrb':
            zrb = du.initZrb()
            zrb = sim_importer.loadZrbData(zrb)
        elif runType == 'updateZrb':
            with open(os.path.join(gp.wdZrb,'pyObjects','zrb.pkl'), "rb") as input_file:
                zrb = pickle.load(input_file)
        plotDash = False
        zrb = computeIndicators(zrb)
        zrb = sim_importer.loadZrbObj(zrb,gp.simDZrb)
        export2MPVT(zrb,'objectives')
        export2MPVT(zrb,'horizon_indicators',pFilterZrb)
        # setTimingGroups(zrb)
        exportZrb2Dashboard(zrb,pFilterZrb)
        exportWs2Geop(zrb,pathwFilter=pFilterZrb)
        # export365ts(zrb,pathwFilter=pFilterZrb)
    ##################################OTB
    if caseStudy['OTB']:
        if runType == 'initOtb':
            otb = du.initOtb()
            otb = sim_importer.loadOtbData(otb)
            otb = sim_importer.loadOtbObj(otb,gp.simDOtb)
            print('OTB loaded')
        elif runType == 'updateOtb':
            with open(os.path.join(gp.dafneParam['OTB']['pklDir'],'otb.pkl'), "rb") as input_file:
                otb = pickle.load(input_file)
                print('OTB loaded')
        otb = computeIndicators(otb)
        export2MPVT(otb,'objectives')
        export2MPVT(otb,'horizon_indicators',pFilterOtb )
        import pdb; pdb.set_trace()
        exportOtb2Dashboard(otb,pFilterOtb )
        exportWs2Geop(otb,pFilterOtb)
    print('calc indicator: THE END')
