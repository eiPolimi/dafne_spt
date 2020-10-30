'''
Dafne Project - Simulation Processing Toolbox
=========================
:title:     DAFNE Utils
:authors:   MarcoMicotti@polimi
:content:   Translates the system model into Python classes of components, variables and indicators, ready to be populated.
            It also defines a number of utility functions, imported by the other script of the toolbox.
'''

import sys
import os
import datetime as dt
import pandas as pd
import pdb
sys.path.insert(0,'PATH/TO/DAFNE/TOOLBOX/HERE/' )
import globalparam as gp

runDebug = True
###############################
#classes
###############################
class Component:
    """Common base class for all components"""
    def __init__(self, name, acronym, type, watersystemAcr,subbasin,country, marker,action_id,variableList=None,featureList=None,indicatorList=None,freq=None):
        self.name      = name
        self.acronym   = acronym
        self.type      = type
        self.system    = watersystemAcr
        self.subbasin  = subbasin
        self.country   = country
        self.marker    = marker
        self.action    = action_id
        self.variables = [createVar(v,freq) for v in variableList ]
        self.features  = [createFeat(f,freq) for f in featureList]
        self.indicators= []

    def getVariable(self,label):
        """given the variable label, return it from the list in the component"""
        try:
            labList   = [i.label for i in self.variables]
            idx       = labList.index(label)
            variable = self.variables[idx]
            return variable,idx
        except:
            print('{}: No variable {} match'.format(self.name,label))
            return None,None

    def getFeature(self,label):
        """given the feature label, return it from the list in the component"""
        try:
            labList   = [f.label for f in self.features]
            idx       = labList.index(label)
            feature = self.features[idx]
            return feature,idx
        except:
            print('{}: No feature {} match'.format(self.name,label))
            return None,None

    def getIndicator(self,label):
        """given the indicator label, return it from the list in the component"""
        try:
            labList   = [i.label for i in self.indicators]
            idx       = labList.index(label)
            indicator = self.indicators[idx]
            return indicator,idx
        except:
            print('{}: No indicator {} match'.format(self.name,label))
            return None,None

    def export2Geop(self,pathwFilter,varFilter=[],indFilter=[]):
        """export all component variables and indicators to a csv file, for further Geoportal import"""
        if varFilter:
            varList    = [v for v in self.variables if v.label in varFilter]
        else:
            varList    = self.variables
        if indFilter:
            indList    = [i for i in self.indicators if i.label in indFilter]
        else:
            indList    = self.indicators
        tsList = varList + indList
        #set dictionary
        pDict = gp.dafneParam[self.system]
        #
        for ts in tsList:
            if ts.data.empty:
                print('.......{} - {} skipped: no data'.format(self.name,ts.label))
            else:
                print('{} - {} starting export...'.format(self.name,ts.label))
                pFilter = pathwFilter.copy()
                if type(ts.data.index) != pd.core.indexes.datetimes.DatetimeIndex:
                    ts.data = setIndTimeIndex(ts.data,pDict['tStart'].year,ts.frequency)
                t1ML = [int(datetime2matlabdn(x)) for x in ts.data.index]
                ts.data.insert(0,'timestart',t1ML)
                if 'timestart' not in pFilter: pFilter.insert(0,'timestart')
                #
                if ts.label in ('h','th','s'):
                    t_sampling = 'instantaneous'
                elif (self.acronym == 'TUR') & (ts.label in ['fy','fd','th']):
                    t_sampling = 'instantaneous'
                else:
                    t_sampling = 'interval'
                    if ts.type != 'Indicator':
                        if self.system == 'ZRB':
                            t2   = pd.date_range(ts.data.index.min(),ts.data.index.max()+ dt.timedelta(weeks=6),freq = pDict['timestep'])
                        elif self.system == 'OTB':
                            t2   = pd.date_range(ts.data.index.min(),ts.data.index.max(),freq = pDict['timestep'])
                        t2ML = [int(datetime2matlabdn(x)) for x in t2]
                        ts.data.insert(1,'timeend',t2ML)
                        if 'timeend' not in pFilter: pFilter.insert(1,'timeend')
                ## NOTE: here a dictionary cannot be used because it is not ordered
                metaLabels = ['name','author','created',
                            'description', 'unit','origin',
                            'time_sampling','frequency','scenario',
                            'marker_id','id_type','interpolator']
                #description & id_type
                if ts.type == 'Indicator':
                    id_type     = 'I'
                    descr       = ts.descr + ' at ' + self.name + ' ' + self.type
                else:
                    id_type     = 'TS_' + ts.label
                    descr       = ts.type + ' at ' + self.name + ' ' + self.type
                #
                metaValues  = [ts.label + '_' + self.acronym, gp.authorName, pd.to_datetime('today'),
                              descr, ts.unit, gp.dafneParam[self.system]['origin'],
                              t_sampling, ts.frequency, pDict['scen'],
                              self.marker, id_type,None]
                outFileName = os.path.join(pDict['tsDir'], metaValues[0] + '.csv')
                dfHead      = pd.DataFrame(metaValues,index = metaLabels)
                dfHead.to_csv(outFileName ,header=False)
                if pFilter:
                    dfOut = ts.data.reindex(columns=pFilter)
                    dfOut.fillna(gp.geop_no_value,inplace = True)
                    dfOut.to_csv(outFileName ,mode='a', index=False, header=True)
                else:
                    ts.data.fillna(gp.geop_no_value,inplace = True)
                    ts.data.to_csv(outFileName ,mode='a', index=False, header=True)
                print('{} - {}: file {} exported!'.format(self.name,ts.type,outFileName))

class SysModel:
    """class to store all the components, with acronym list for selection"""
    def __init__(self,name,acronym,componentList):
        self.name       = name
        self.acronym    = acronym
        self.components = componentList
        self.acronyms   = [c.acronym for c in self.components]
        self.indicators = []

    def getComponent(self,acronym):
        """given the component acronym, return a component from the system model and its index"""
        try:
            idx       = self.acronyms.index(acronym)
            component = self.components[idx]
            return component,idx
        except:
            print('No component match')
            return None


class Variable:
    def __init__(self, label, type, unit, freq, values = None):
       self.label     = label
       self.type      = type
       self.unit      = unit
       self.frequency = freq
       if values is None:
            values = pd.DataFrame()
       self.data      = values

class Indicator(Variable):
    def __init__(self, label,unit,freq,sector,descr,stat,suffix,link,fundir='minimize',values = None):
        Variable.__init__(self, label, 'Indicator', unit,freq, values)
        self.sector        = sector
        self.descr         = descr
        ## to be implemented for ZRB
        # self.subbasin      = subbasin
        # self.country       = country
        ##
        self.stat          = stat
        self.suffix        = suffix
        self.funDirection  = fundir
        self.link          = link

    def plotminmaxVF(self, saveDir,inverse=False):
        """plot standard linear value function using min/max values of the dataset"""
        if inverse:
            xValues = [0, -df.data.max(), - df.data.min(),- df.data.min()*1.25]
        else:
            xValues = [0, df.data.min(), df.data.max(), df.data.max()*1.25]
        if self.funDirection == 'maximize':
            yValues = [ 0, 0, 1, 1]
        elif self.funDirection == 'minimize':
            yValues = [ 1, 1, 0, 0]
        else:
            print ('ERROR!! Unknown value for funDirection ')
        #
        ax      = plt.plot(xValues,yValues)
        plt.xlabel('{} {}'.format(self.descr, self.unit))
        plt.ylabel('satisfaction [-]')
        # plt.show(block=False)
        plt.savefig(os.path.join(saveDir,self.label + 'minmaxVF.png'))
        plt.close()


###############################
#functions
###############################
def intersectList(l1,l2):
    return list(set(l1).intersection(l2))

def datetime2matlabdn(dtime):
    '''taken from: https://stackoverflow.com/questions/8776414/python-datetime-to-matlab-datenum#8776555'''
    ord = dtime.toordinal()
    mdn = dtime + dt.timedelta(days = 366)
    frac = (dtime - dt.datetime(dtime.year,dtime.month,dtime.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
    return mdn.toordinal() + frac

def createVar(varLabel,freq):
    if varLabel == 'a':
        var = Variable(varLabel,'inflow','[m3/sec]',freq)
    elif varLabel == 'h':
        var = Variable(varLabel,'water level','[m]',freq)
    elif varLabel == 'th':
        var = Variable(varLabel,'target water level','[m]',freq)
    elif varLabel == 's':
        var = Variable(varLabel,'storage','[Mm3]',freq)
    elif varLabel == 'r':
        var = Variable(varLabel,'release','[m3/sec]',freq)
    elif varLabel == 'tr':
        var = Variable(varLabel,'turbined release','[m3/sec]',freq)
    elif varLabel == 'q':
        var = Variable(varLabel,'flow','[m3/sec]',freq)
    elif varLabel == 'mef':
        var = Variable(varLabel,'minimum environmental flow','[m3/sec]',freq)
    elif varLabel == 'tf':
        var = Variable(varLabel,'target flow','[m3/sec]',freq)
    elif varLabel == 'en_mwh':
        var = Variable(varLabel,'energy production','[MWh]',freq)
    elif varLabel == 'ed_mwh':
        var = Variable(varLabel,'energy demand','[MWh]',freq)
    elif varLabel == 'en_gwh':
        var = Variable(varLabel,'energy production','[GWh]',freq)
    elif varLabel == 'ed_gwh':
        var = Variable(varLabel,'energy demand','[GWh]',freq)
    elif varLabel == 'en_twh':
        var = Variable(varLabel,'energy production','[TWh]',freq)
    elif varLabel == 'ed_twh':
        var = Variable(varLabel,'energy demand','[TWh]',freq)
    elif varLabel == 'wd':
        var = Variable(varLabel,'water demand','[m3/sec]',freq)
    elif varLabel == 'nwd':
        var = Variable(varLabel,'nominal water demand','[m3/sec]',freq)
    elif varLabel == 'swd':
        var = Variable(varLabel,'simulated water demand','[m3/sec]',freq)
    elif varLabel == 'wa':
        var = Variable(varLabel,'water abstraction','[m3/sec]',freq)
    elif varLabel == 'd':
        var = Variable(varLabel,'deficit','[m3/sec]',freq)
    elif varLabel == 'fy':
        var = Variable(varLabel,'fish yield','[MT]',freq)
    elif varLabel == 'fd':
        var = Variable(varLabel,'fish yield deficit','[MT]',freq)
    elif varLabel == 'ws':
        var = Variable(varLabel,'water surface','[m2]',freq)
    else:
        var = 'Unknown type'
    return var

def createFeat(featLabel,freq):
    if featLabel == 'Power':
        feat = Variable(featLabel,'Installed power','[MW]',freq)
    elif featLabel == 'Area':
        feat = Variable(featLabel,'Irrigated area','[ha]',freq)
    elif featLabel == 'Flow':
        feat = Variable(featLabel,'Minimum flow requirement','[m3/sec]',freq)
    elif featLabel == 'Water level':
        feat = Variable(featLabel,'Minimum water level requirement','[m]',freq)
    else:
        feat = 'Unknown type'
    return feat

def setIndTimeIndex(df,firstY,period):
    if period == 'daily' and df.index.size == 365:
        tStart  = dt.datetime(firstY,1,1)
        tEnd    = dt.datetime(firstY,12,31)
        tIndex  = pd.date_range(tStart,tEnd,freq = 'D')
        df.set_index(tIndex,inplace = True)
    elif period == 'monthly' and df.index.size == 12:
        tStart  = dt.datetime(firstY,1,1)
        tEnd    = dt.datetime(firstY,12,1)
        tIndex  = pd.date_range(tStart,tEnd,freq = 'MS')
        df.set_index(tIndex,inplace = True)
    elif period == 'yearly':
        tStart  = dt.datetime(df.index.min(),1,1)
        tEnd    = dt.datetime(df.index.max(),1,1)
        tIndex  = pd.date_range(tStart,tEnd,freq = 'YS')
        df.set_index(tIndex,inplace = True)
    elif period == 'horizon':
        tStart  = dt.datetime(firstY,1,1)
        df.set_index(pd.DatetimeIndex([tStart]),inplace = True)
    else:
        print('Check time index: no cyclostationary time period identified')
    return df

def minmaxNorm(df,funDirList,flip= False,append=False):
    '''Normalization of dataset wrt the range of its values. NOTE: df must have pathways on the columns and indicators on the rows..'''
    dfN     = pd.DataFrame(index=df.index, columns=df.columns)
    if flip:
        for i,r in df.iterrows():
            idx = df.index.to_list().index(i)
            if funDirList[idx] == 'minimize':
                if (r.max() - r.min()) == 0:
                    dfN.loc[i,:] = 0
                else:
                    dfN.loc[i,:]     = (r.max() - r)/(r.max() - r.min())
            elif funDirList[idx] == 'maximize':
                if (r.max() - r.min()) == 0:
                    dfN.loc[i,:] = 0
                else:
                    dfN.loc[i,:]     = (r - r.min())/(r.max() - r.min())
            else:
                print('ERROR: Unknown funDirection {}'.format(funDirList[i]))
                return
    else:
        dfN        = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()),axis=1)
    if append:
        dfN.columns= [c.replace('P','Pn') for c in df.columns]
        dfOut      = pd.concat([df,dfN],axis=1)
    else:
        dfOut      = dfN
    return dfOut


###############################
#init components Zambezi
###############################
def initZrb():
    """Initialise Components and Sytem Model for the Zambezi case studies. It returns a dictionary of pathways"""
    #existing reservoir/power plants
    Vic = Component('Victoria Falls','VIC', 'Run of the River Hydro Power Plant','ZRB','','','rhp_VIC',['a_E_1'],['tr','en_twh','ed_twh'], ['Power'],freq=gp.freqZrb)
    Kar = Component('Kariba','KAR', 'Reservoir and Hydro Power Plant','ZRB','','','rhp_KAR',['a_E_2'],['a','h','s','r','tr','en_twh','ed_twh'], ['Power'],freq=gp.freqZrb)
    Itt = Component('Itezhi-Tezhi','ITT', 'Reservoir and Hydro Power Plant','ZRB','','','rhp_ITT',['a_E_3'],['a','h','s','r','tr','en_twh','ed_twh'], ['Power'],freq=gp.freqZrb)
    Kgu = Component('Kafue Gorge Upper','KGU', 'Reservoir and Hydro Power Plant','ZRB','','','rhp_KGU',['a_E_4'],['a','h','s','r','tr','en_twh','ed_twh'], ['Power'],freq=gp.freqZrb)
    CaB = Component('Cahora Bassa','CAB', 'Reservoir and Hydro Power Plant','ZRB','','','rhp_CAB',['a_E_5'],['a','h','s','r','tr','en_twh','ed_twh'], ['Power'],freq=gp.freqZrb)

    #planned reservoirs
    Kgl = Component('Kafue Gorge Lower','KGL', 'Reservoir and Hydro Power Plant','ZRB','','','rhp_KGL',['a_E_6'],['a','h','s','r','tr','en_twh','ed_twh'], ['Power'],freq=gp.freqZrb)
    Bat = Component('Batoka','BAT', 'Reservoir and Hydro Power Plant','ZRB','','','rhp_ITT',['a_E_7'],['a','h','s','r','tr','en_twh','ed_twh'], ['Power'],freq=gp.freqZrb)
    Dev = Component('Devil Gorge','DEV', 'Reservoir and Hydro Power Plant','ZRB','','','rhp_DEV',['a_E_8'],['a','h','s','r','tr','en_twh','ed_twh'], ['Power'],freq=gp.freqZrb)
    Mnk = Component('Mphanda Nkuwa','MNK', 'Reservoir and Hydro Power Plant','ZRB','','','rhp_MNK',['a_E_10'],['a','h','s','r','tr','en_twh','ed_twh'], ['Power'],freq=gp.freqZrb)
    #other planned reservoirs not considered in the final version of DAFNE model
    # Mup = Component('Mupata Gorge','MUP', 'Reservoir and Hydro Power Plant','ZRB','','','rhp_MUP',['a_E_'],[], ['Power'],freq=gp.freqZrb)
    # Bor = Component('Boroma','BOR', 'Reservoir and Hydro Power Plant','ZRB','','','rhp_BOR',['a_E_'],[], ['Power'],freq=gp.freqZrb)
    # Lup = Component('Lupata','LUP', 'Reservoir and Hydro Power Plant','ZRB','','','rhp_LUP',['a_E_'],[], ['Power'],freq=gp.freqZrb)

    #planned copper mine - not considered
    # Scm = Component('Synclinorium','SCM', 'Copper Mine','ZRB','','','cm_SCM',['a_SE_14'],[], [],freq=gp.freqZrb)

    #Equivalent Irrigation District
    Ir2 = Component('Equivalent Irrigation District no. 2, downstream Batoka Gorge','ID2', 'Irrigation District','ZRB','','','irr_ID2',['a_F_20','a_F_49'],['wd','wa','d'], ['Area'],freq=gp.freqZrb)
    Ir3 = Component('Equivalent Irrigation District no. 3, downstream Kariba','ID3', 'Irrigation District','ZRB','','','irr_ID3',['a_F_19', 'a_F_46','a_F_48'],['wd','wa','d'], ['Area'],freq=gp.freqZrb)
    Ir4 = Component('Equivalent Irrigation District no. 4, downstream Ithezi Ithezi','ID4', 'Irrigation District','ZRB','','','irr_ID4',['a_F_41'],['wd','wa','d'], ['Area'],freq=gp.freqZrb)
    Ir5 = Component('Equivalent Irrigation District no. 5, downstream Kafue Gorge Lower','ID5', 'Irrigation District','ZRB','','','irr_ID5',['a_F_17'],['wd','wa','d'], ['Area'],freq=gp.freqZrb)
    Ir6 = Component('Equivalent Irrigation District no. 6, downstream Kafue-Zambezi confluence','ID6', 'Irrigation District','ZRB','','','irr_ID6',['a_F_16','a_F_44'],['wd','wa','d'], ['Area'],freq=gp.freqZrb)
    Ir7 = Component('Equivalent Irrigation District no. 7, downstream Cahora Bassa','ID7', 'Irrigation District','ZRB','','','irr_ID7',[],['wd','wa','d'], ['Area'],freq=gp.freqZrb)
    Ir8 = Component('Equivalent Irrigation District no. 8, downstream Mphanda Nkuwa','ID8', 'Irrigation District','ZRB','','','irr_ID8',['a_F_9','a_F_36'],['wd','wa','d'], ['Area'],freq=gp.freqZrb)
    Ir9 = Component('Equivalent Irrigation District no. 9, downstream Shire-Zambezi confluence','ID9', 'Irrigation District','ZRB','','','irr_ID9',['a_F_2','a_F_30'],['wd','wa','d'], ['Area'],freq=gp.freqZrb)

    #Ecosystem Protection Area
    Zde = Component('Zambezi Delta','ZDE', 'Ecosystem Protection Area','ZRB','','','env_ZDE',['a_W_Ec_13'],['q','tf'], ['Flow'],freq=gp.freqZrb)

    #Ecosystem Protection Area - existing not considered
    # Vfa = Component('Victoria Falls','VFA', 'Wetland','ZRB','','','env_VFA',['a_W_Ec_XX'],['q','mef'], ['Flow'],freq=gp.freqZrb)

    #Ecosystem Protection Area - planned not considered
    # Kfl = Component('Kafue Flats','KFL', 'Ecosystem Protection Area','ZRB','','','env_KFL',['a_W_Ec_13'],[], ['Flow'],freq=gp.freqZrb)
    # Zmz = Component('Zimbabwe-Mozambique-Zambia (ZIMOZA)','ZMZ', 'Ecosystem Protection Area','ZRB','','','env_ZMZ',['a_W_Ec_19'],[], ['Flow'],freq=gp.freqZrb)
    # Lzm = Component('Lower Zambezi-Mana Pools','LZM', 'Ecosystem Protection Area','ZRB','','','env_LZM',['a_W_Ec_20'],[], ['Flow'],freq=gp.freqZrb)
    # Lpm = Component('Liuwa Plains-Mussuma','LPM', 'Ecosystem Protection Area','ZRB','','','env_LPM',['a_W_Ec_21'],[], ['Flow'],freq=gp.freqZrb)
    return SysModel('Zambezi River Basin', 'ZRB',[Itt, Kgu, Kar, CaB, Vic, Bat, Kgl, Dev,Mnk, Ir2, Ir3, Ir4, Ir5, Ir6, Ir7, Ir8,Ir9, Zde])


def initOtb():
    """Initialise Components and Sytem Model for the Omo-Turana case study. It returns a SysModel object"""
    #existing reservoir/power plants
    G1  = Component('Gibe1','G1', 'Reservoir and Hydro Power Plant','OTB','Omo','Ethiopia','rhp_G1',[],['a','h','s','r','tr','en_gwh','ed_gwh'], ['Power'],freq=gp.freqOtb)
    G2  = Component('Gibe2','G2', 'Hydro Power Plant','OTB','Omo','Ethiopia','rhp_G2',[],['r','tr','en_gwh','ed_gwh'], ['Power'],freq=gp.freqOtb)
    G3  = Component('Gibe3','G3', 'Reservoir and Hydro Power Plant','OTB','Omo','Ethiopia','rhp_G3',[],['a','h','s','r','tr','ws','en_gwh','ed_gwh'], ['Power'],freq=gp.freqOtb)
    Koy = Component('Koysha','KOY', 'Reservoir and Hydro Power Plant','OTB','Omo','Ethiopia','rhp_KOY',[],['a','h','s','r','tr','ws','en_gwh','ed_gwh'], ['Power'],freq=gp.freqOtb)

    #Irrigation District
    Ir1 = Component('Kuraz Irrigation Scheme','IR1', 'Irrigation District','OTB','Omo','Ethiopia','irr_ID1',[],['nwd','swd','wa','d'], ['Area'],freq=gp.freqOtb)
    Ir2 = Component('Private Irrigation Scheme, lower Omo','IR2', 'Irrigation District','OTB','Omo','Ethiopia','irr_ID2',[],['nwd','swd','wa','d'], ['Area'],freq=gp.freqOtb)
    Rec = Component('Recession Agriculture area, lower Omo','REC', 'Irrigation District','OTB','Omo','Ethiopia','irr_REC',[],['q','tf'], ['Area'],freq=gp.freqOtb)

    #Ecosystem Protection Area - existing
    Ode = Component('Omo Delta','ODE', 'Ecosystem Protection Area','OTB','Omo','Ethiopia','env_ODE',[],['q','tf'], ['Flow'],freq=gp.freqOtb)
    Tur = Component('Lake Turkana','TUR', 'Natural lake','OTB','Turkana','Kenya','env_TUR',[''],['h','th','fy','fd'], ['Water level'],freq=gp.freqOtb)

    return SysModel('Omo-Turkana River Basin', 'OTB',[G1,G2,G3,Koy,Ir1,Ir2,Rec,Ode,Tur])
