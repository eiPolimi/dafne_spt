"""
DAFNE Project - Simulation Processing Toolbox
---------------------------------------------
:title:     Indicators library
:authors:   MarcoMicotti@polimi
:content:   Library of Indicators for Food, Energy and Water-Ecosystem sectors

"""
import os
import sys
import pandas as pd
import numpy as np
sys.path.insert(0,'PATH/TO/DAFNE/TOOLBOX/HERE/' )
import dafneutils as du
import pdb
################################################################################
# init
################################################################################
runDebug = True

################################################################################
# functions
################################################################################
def periodicStat(df,period,statFun):
    '''Given a DataFrame, group it by the given period, extract a statistic and return a new dataframe'''
    if period == 'daily':
        # if runDebug: pdb.set_trace()
        pKey        = lambda x: x.dayofyear
    elif period == 'monthly':
        pKey        = lambda x: x.month
    elif period == 'yearly':
        pKey        = lambda x: x.year
    elif period == 'horizon':
        df['']      = 0
        pKey        = ''
    # elif period == 'doy':
    #     pKey        = lambda x: x.year
    # tip: %j Day of the year as a zero-padded decimal number.001, 002, …, 366
    else:
        print('Unknown period')
    dfGr        = df.groupby(pKey)
    statFun2Eval= ('df.' + '().'.join(statFun.split('.'))+'()').replace('df','dfGr')
    try:
        dfStat      = eval(statFun2Eval)
        # remove cyclostationary mean on day 366
        if (period == 'daily') & (366 in dfStat.index):
            dfStat.drop(labels=366,axis=0,inplace=True)
        # in case of combined statistic, dfStat became a pd.Series, but this is not coherent, so here it change its type back to DataFrame
        if type(dfStat) == type(pd.Series()):
            dfStat = pd.DataFrame(dfStat)
        return dfStat
    except:
        if runDebug: pdb.set_trace()
        print('wrong stat, check better your code!')

################################################################################
# indicators functions
################################################################################

def i_en_production_m_mean(df,uom):
    i               = du.Indicator(label='i_E_2_EnProd_MeanM', unit='{}/m'.format(uom),sector ='Energy', descr='Average monthly energy production',
                                    freq='monthly',stat='mean',suffix='_mean_m',link='iexxx',fundir='maximize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_production_y_sum(df,uom):
    i               = du.Indicator(label='i_E_2_EnProd_SumY', unit='{}/y'.format(uom),sector ='Energy', descr='Yearly amount of energy production',
                                    freq='yearly',stat='sum',suffix='_sum_y',link='iexxx',fundir='maximize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_production_sum(df,uom):
    i               = du.Indicator(label='i_E_2_EnProd_Sum', unit='{}'.format(uom),sector ='Energy', descr='Total amount of Energy production',
                                    freq='horizon',stat='sum',suffix='_sum',link='iexxx',fundir='maximize')
    # if runDebug: pdb.set_trace()
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_production_d_mean(df,uom):
    i               = du.Indicator(label='i_E_2_EnProd_MeanD', unit='{}'.format(uom),sector ='Energy', descr='Daily average Energy production',
                                    freq='horizon',stat='mean',suffix='_mean',link='iexxx',fundir='maximize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_production_y_mean(df,uom):
    i               = du.Indicator(label='i_E_2_EnProd_MeanY', unit='{}'.format(uom),sector ='Energy', descr='Yearly average Energy production',
                                    freq='horizon',stat='mean',suffix='_mean',link='iexxx',fundir='maximize')
    df              = df.resample('Y').sum()
    #remove years before dam building from the average
    df.replace(0.0,np.nan,inplace=True)
    # if runDebug: pdb.set_trace()
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_demand_y_sum(df,uom):
    i               = du.Indicator(label='i_E_EnDemand_SumY', unit='{}/y'.format(uom),sector ='Energy', descr='Energy demand',
                                    freq='yearly',stat='sum',suffix='_sum_y',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_deficit_m_mean(df,uom):
    i               = du.Indicator(label='i_E_3_EnDef_MeanM', unit='{}/m'.format(uom),sector ='Energy', descr='Energy deficit',
                                    freq='monthly',stat='mean',suffix='_mean_m',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_deficit_y_sum(df,uom):
    i               = du.Indicator(label='i_E_3_EnDef_SumY', unit='{}/y'.format(uom),sector ='Energy', descr='Energy deficit',
                                    freq='yearly',stat='sum',suffix='_sum_y',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_deficit_y_mean(df,uom):
    i               = du.Indicator(label='i_E_3_EnDef_MeanY', unit='{}/y'.format(uom),sector ='Energy', descr='Yearly energy deficit',
                                    freq='horizon',stat='mean',suffix='_mean',link='iexxx',fundir='minimize')
    df              = df.resample('Y').sum()
    df.replace(0.0,np.nan,inplace=True)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_deficit_sum(df,uom):
    i               = du.Indicator(label='i_E_3_EnDef_Sum', unit='{}/y'.format(uom),sector ='Energy', descr='Energy deficit',
                                    freq='horizon',stat='sum',suffix='_sum',link='iexxx',fundir='minimize')

    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_wlcyclo_mean(df,vLabel):
    i               = du.Indicator(label='i_E_HCycloMean_{}'.format(vLabel), unit=['m'],sector ='Energy', descr='Water level cyclostationary mean',
                                    freq='daily',stat='mean',suffix='_mean',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_wlcyclo_min(df,vLabel):
    i               = du.Indicator(label='i_E_HCycloMin_{}'.format(vLabel), unit=['m'],sector ='Energy', descr='Water level cyclostationary minimum',
                                    freq='daily',stat='min',suffix='_min',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_wlcyclo_max(df,vLabel):
    i               = du.Indicator(label='i_E_HCycloMax_{}'.format(vLabel), unit=['m'],sector ='Energy', descr='Water level cyclostationary maximum',
                                    freq='daily',stat='max',suffix='_max',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_wlcyclo_m_mean(df,vLabel):
    i               = du.Indicator(label='i_E_HCycloMeanM_{}'.format(vLabel), unit=['m'],sector ='Energy', descr='Water level cyclostationary mean',
                                    freq='monthly',stat='mean',suffix='_mean_m',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_wlcyclo_m_max(df,vLabel):
    i               = du.Indicator(label='i_E_HCycloMaxM_{}'.format(vLabel), unit=['m'],sector ='Energy', descr='Water level cyclostationary max',
                                    freq='monthly',stat='max',suffix='_max_m',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_wlcyclo_m_min(df,vLabel):
    i               = du.Indicator(label='i_E_HCycloMinM_{}'.format(vLabel), unit=['m'],sector ='Energy', descr='Water level cyclostationary min',
                                    freq='monthly',stat='min',suffix='_min_m',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_scyclo_m_mean(df):
    i               = du.Indicator(label='i_E_SCycloMeanM', unit=['m3'],sector ='Energy', descr='Storage cyclostationary mean',
                                    freq='monthly',stat='mean',suffix='_mean_m',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_rcyclo_mean(df):
    i               = du.Indicator(label='i_E_RCycloMean', unit=['m3/sec'],sector ='Energy', descr='Release cyclostationary mean',
                                    freq='daily',stat='mean',suffix='_mean_d',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_rcyclo_m_mean(df):
    i               = du.Indicator(label='i_E_RCycloMeanM', unit=['m3/sec'],sector ='Energy', descr='Release cyclostationary mean',
                                    freq='monthly',stat='mean',suffix='_mean_m',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_en_firstoperatingyear(df):
    i               = du.Indicator(label='i_E_year1EnProd', unit=['year'],sector ='Energy', descr='First year of operation',
                                    freq='horizon',stat='',suffix='',link='iexxx',fundir='maximize')
    dfBool          = df>0
    i.data          = pd.DataFrame(dfBool.idxmax().apply(lambda x: x.year)).T
    return i
#
def i_en_firstoperatingday(df,r):
    i               = du.Indicator(label='i_E_day1EnProd', unit=['time'],sector ='Energy', descr='Start of operation',
                                    freq='horizon',stat='',suffix='',link='iexxx',fundir='maximize')
    # if runDebug: pdb.set_trace()
    dfBool          = df>0
    i.data          = pd.DataFrame(dfBool.idxmax()).T
    # nan management
    idx = df.isna().all()
    i.data.loc[:,idx.to_numpy()] = np.nan
    return i
#
##################
#IRRIGATION
##################
def i_irr_deficit_m_mean(df):
    i               = du.Indicator(label='i_F_IrrDef_MeanM', unit=['m3/sec'],sector ='Food', descr='Water deficit for irrigation',
                                    freq='monthly',stat='mean',suffix='_mean_m',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_deficit_y_sum(df):
    i               = du.Indicator(label='i_F_IrrDef_SumY', unit=['m3/sec'],sector ='Food', descr='Water deficit for irrigation',
                                    freq='yearly',stat='sum',suffix='_sum_y',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_deficit_sum(df):
    i               = du.Indicator(label='i_F_IrrDef_Sum', unit=['m3/sec'],sector ='Food', descr='Water deficit for irrigation',
                                    freq='horizon',stat='sum',suffix='_sum',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_deficit_y_mean(df,dataFreq):
    i               = du.Indicator(label='i_F_IrrDef_YMean', unit=['Mm3'],sector ='Food', descr='Yearly average volume of water deficit for irrigation',
                                    freq='horizon',stat='mean',suffix='_sum',link='iexxx',fundir='minimize')
    # unit of measurement conversion
    if dataFreq == 'daily':
        df          = df*86400/(10**6)
    elif dataFreq == 'monthly':
        df          = df*86400*30.41/(10**6) #365/12
        print('# WARNING: from m3/sec to Mm3... to be improved')
    #sum yearly deficit
    df              = df.resample('Y').sum()
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_distancepos_y_d_mean(df):
    # if runDebug: pdb.set_trace()
    i               = du.Indicator(label='i_F_IrrDistPos_DMeanY', unit=['m3/sec'],sector ='Food', descr='Daily positive distance from artificial flood values',
                                    freq='yearly',stat='mean',suffix='_mean_y',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_distancepos_mean(df):
    i               = du.Indicator(label='i_F_IrrDistPos_DMean', unit=['m3/sec'],sector ='Food', descr='Daily positive distance from artificial flood values',
                                    freq='horizon',stat='mean',suffix='_mean',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_distanceneg_y_d_mean(df):
    i               = du.Indicator(label='i_F_IrrDistNeg_DMeanY', unit=['m3/sec'],sector ='Food', descr='Daily negative distance from artificial flood values',
                                    freq='yearly',stat='mean',suffix='_mean_y',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_distanceneg_mean(df):
    i               = du.Indicator(label='i_F_IrrDistNeg_DMean', unit=['m3/sec'],sector ='Food', descr='Daily negative distance from artificial flood values',
                                    freq='horizon',stat='mean',suffix='_sum',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_abstraction_m_mean(df,vLabel):
    i               = du.Indicator(label='i_F_IrrMeanM_{}'.format(vLabel), unit=['m3/sec'],sector ='Food', descr='Water abstraction for irrigation',
                                    freq='monthly',stat='mean',suffix='_mean',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i

def i_irr_abstraction_y_sum(df,dataFreq):
    i               = du.Indicator(label='i_F_IrrWa_SumY', unit=['Mm3'],sector ='Food', descr='Volume of water abstraction for irrigation',
                                    freq='yearly',stat='sum',suffix='_sum',link='iexxx',fundir=None)
    # unit of measurement conversion
    if dataFreq == 'daily':
        df          = df*86400/(10**6)
    elif dataFreq == 'monthly':
        df          = df*86400*30.41/(10**6) #365/12
        print('# WARNING: from m3/sec to Mm3... to be improved')
    #
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_abstraction_y_mean(df,dataFreq):
    i               = du.Indicator(label='i_F_IrrWa_MeanY', unit=['Mm3'],sector ='Food', descr='Yearly Volume of water abstraction for irrigation',
                                    freq='horizon',stat='mean',suffix='_mean',link='iexxx',fundir='maximize')
    # unit of measurement conversion
    if dataFreq == 'daily':
        df          = df*86400/(10**6)
    elif dataFreq == 'monthly':
        df          = df*86400*30.41/(10**6) #365/12
        print('# WARNING: from m3/sec to Mm3... to be improved')
    #
    df              = df.resample('Y').sum()
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_wdemand_y_sum(df,dataFreq):
    i               = du.Indicator(label='i_F_IrrWd_SumY', unit=['Mm3'],sector ='Food', descr='Volume of water requirement for irrigation',
                                    freq='yearly',stat='sum',suffix='_sum',link='iexxx',fundir=None)
    if dataFreq == 'daily':
        df          = df*86400/(10**6)
    elif dataFreq == 'monthly':
        df          = df*86400*30.41/(10**6) #365/12
        print('# WARNING: from m3/sec to Mm3... to be improved')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_fdeficit_y(df):
    i               = du.Indicator(label='i_F_IrrFishDef_Y', unit=['MT/y'],sector ='Food', descr='Fish Yield deficit',
                                    freq='yearly',stat='sum',suffix='_sum_y',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_fdeficit_sum(df):
    i               = du.Indicator(label='i_F_IrrFishDef_Sum', unit=['MT'],sector ='Food', descr='Fish Yield deficit',
                                    freq='horizon',stat='sum',suffix='_sum',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_flake_y(df):
    i               = du.Indicator(label='i_F_IrrFishProd_Y', unit=['MT'],sector ='Food', descr='Fish Yield',
                                    freq='yearly',stat='sum',suffix='_sum_y',link='iexxx',fundir='maximize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_flake_sum(df):
    i               = du.Indicator(label='i_F_IrrFishProd_Sum', unit=['MT'],sector ='Food', descr='Fish Yield',
                                    freq='horizon',stat='sum',suffix='_sum',link='iexxx',fundir='maximize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_fishres_y(df):
    i               = du.Indicator(label='i_F_IrrFishRes_Y', unit=['MT/y'],sector ='Food', descr='Fish Yield in Reservoir',
                                    freq='yearly',stat='sum',suffix='_sum_y',link='iexxx',fundir='maximize')
    #in this case df contains already indicators data
    i.data          = df
    return i
#
def i_irr_fishres_sum(df):
    i               = du.Indicator(label='i_F_IrrFishRes_Sum', unit=['MT'],sector ='Food', descr='Fish Yield in Reservoir',
                                    freq='horizon',stat='sum',suffix='_sum',link='iexxx',fundir='maximize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_fishres_mean(df):
    i               = du.Indicator(label='i_F_IrrFishRes_Mean', unit=['MT/y'],sector ='Food', descr='Yearly average Fish Yield in Reservoir',
                                    freq='horizon',stat='mean',suffix='_sum',link='iexxx',fundir='maximize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_qcyclo_mean(df,vLabel):
    i               = du.Indicator(label='i_F_QCycloMean_{}'.format(vLabel), unit=['m3/sec'],sector ='Food', descr='Cyclostationary mean of the streamflow',
                                    freq='daily',stat='mean',suffix='_mean',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_qcyclo_max(df,vLabel):
    i               = du.Indicator(label='i_F_QCycloMax_{}'.format(vLabel), unit=['m3/sec'],sector ='Food', descr='Cyclostationary maximum of the streamflow',
                                    freq='daily',stat='max',suffix='_max',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_irr_qcyclo_min(df,vLabel):
    i               = du.Indicator(label='i_F_QCycloMin_{}'.format(vLabel), unit=['m3/sec'],sector ='Food', descr='Cyclostationary minimum of the streamflow',
                                    freq='daily',stat='min',suffix='_min',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
##################
#ENVIRONMENT
##################
def i_env_distancepos_m_mean(df):
    i               = du.Indicator(label='i_W_QEnvDistPos_MeanM', unit=['m3/sec'],sector ='Water-Environment', descr='Streamflow deficit with respect to an environmental target flow',
                                    freq='monthly',stat='mean',suffix='_mean_m',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_env_distancepos_y_mean(df):
    i               = du.Indicator(label='i_W_QEnvDistPos_MeanY', unit=['m3/sec'],sector ='Water-Environment', descr='Streamflow deficit with respect to an environmental target flow',
                                    freq='yearly',stat='mean',suffix='_mean_y',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_env_distancepos_mean(df):
    i               = du.Indicator(label='i_W_QEnvDistPos_Mean', unit=['m3/sec'],sector ='Water-Environment', descr='Streamflow deficit with respect to an environmental target flow',
                                    freq='horizon',stat='mean',suffix='_mean',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
# #
# def i_env_squaredistance_y_mean(df):
#     i               = du.Indicator(label='i_W_QEnvSqDist_MeanY', unit=['(m3/sec)^2'],sector ='Water-Environment', descr='Flow positive distance with respect to an environmental target flow',
#                                     freq='yearly',stat='mean',suffix='_mean',link='iexxx',fundir='minimize')
#     if runDebug: pdb.set_trace()
#     i.data          = periodicStat(df,i.frequency,i.stat)
#     return i
# #
# def i_env_squaredistance_mean(df):
#     i               = du.Indicator(label='i_W_QEnvSqDist_Mean', unit=['(m3/sec)^2'],sector ='Water-Environment', descr='Flow positive distance with respect to an environmental target flow',
#                                     freq='horizon',stat='mean',suffix='_mean',link='iexxx',fundir='minimize')
#     if runDebug: pdb.set_trace()
#     i.data          = periodicStat(df,i.frequency,i.stat)
#     return i
#
def i_env_distanceneg_m_mean(df):
    i               = du.Indicator(label='i_W_QEnvDistNeg_MeanM', unit=['m3/sec'],sector ='Water-Environment', descr='Flow negative distance with respect to an environmental target flow',
                                    freq='monthly',stat='mean',suffix='_mean_m',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_env_distanceneg_y_mean(df):
    i               = du.Indicator(label='i_W_QEnvDistNeg_SumY', unit=['m3/sec'],sector ='Water-Environment', descr='Flow negative distance with respect to an environmental target flow',
                                    freq='yearly',stat='mean',suffix='_mean_y',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_env_distanceneg_mean(df):
    i               = du.Indicator(label='i_W_QEnvDistNeg_Sum', unit=['m3/sec'],sector ='Water-Environment', descr='Flow negative distance with respect to an environmental target flow',
                                    freq='horizon',stat='mean',suffix='_mean',link='iexxx',fundir='minimize')
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_env_qcyclo_m_mean(df,vLabel):
    i               = du.Indicator(label='i_W_QCycloMMean_{}'.format(vLabel), unit=['m3/sec'],sector ='Water-Environment', descr='Cyclostationary mean of the streamflow',
                                    freq='monthly',stat='mean',suffix='_mean',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_env_qcyclo_mean(df,vLabel):
    i               = du.Indicator(label='i_W_QCycloMean_{}'.format(vLabel), unit=['m3/sec'],sector ='Water-Environment', descr='Cyclostationary mean of the streamflow',
                                    freq='daily',stat='mean',suffix='_mean',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_env_qcyclo_max(df,vLabel):
    i               = du.Indicator(label='i_W_QCycloMax_{}'.format(vLabel), unit=['m3/sec'],sector ='Water-Environment', descr='Cyclostationary maximum of the streamflow',
                                    freq='daily',stat='max',suffix='_max',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_env_qcyclo_min(df,vLabel):
    i               = du.Indicator(label='i_W_QCycloMin_{}'.format(vLabel), unit=['m3/sec'],sector ='Water-Environment', descr='Cyclostationary minimum of the streamflow',
                                    freq='daily',stat='min',suffix='_min',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
def i_env_wlcyclo_mean(df):
    i               = du.Indicator(label='i_Env_HCycloMean', unit=['m'],sector ='Water-Environment', descr='Cyclostationary mean of the water level',
                                    freq='daily',stat='mean',suffix='_mean',link='iexxx',fundir=None)
    i.data          = periodicStat(df,i.frequency,i.stat)
    return i
#
