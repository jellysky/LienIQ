import numpy as np
import pandas as pd
import datetime as dtt
import dateutils as du

class const():
    @staticmethod
    def monthsInYear():
        return 12
    @staticmethod
    def borrowRate():
        return .10
    @staticmethod
    def exitStr():
        return ['noteSale','reinState','bk13','fcPayoff','fcREO','cashFlow','loanMod']
    @staticmethod
    def pnlRowMax():
        return 500
    @staticmethod
    def loanBoardingFee():
        return 45
    @staticmethod
    def ownTransferNotice():
        return 25
    @staticmethod
    def standardServicingFee():
        return 15
    @staticmethod
    def specialtyServicingFee():
        return 30
    @staticmethod
    def deBoardingFee():
        return 15
    @staticmethod
    def overnightFee():
        return 40
    @staticmethod
    def bpoCost():
        return 150
    @staticmethod
    def ownEncReportCost():
        return 145
    @staticmethod
    def creditReportCost():
        return 18
    @staticmethod
    def pacerCost():
        return 5
    @staticmethod
    def recordingFee():
        return 25
    @staticmethod
    def wireFee():
        return 30
    @staticmethod
    def rehabCost():
        return 20000
    @staticmethod
    def foreclosureProceeding():
        return 2800
    @staticmethod
    def eviction():
        return 1100
    @staticmethod
    def noteBuyHaircut():
        return .7
    @staticmethod
    def noteSaleHaircut():
        return .8
    @staticmethod
    def noteSaleMonthsAhead():
        return 3
    @staticmethod
    def reinStateMonthsAhead():
        return 3
    @staticmethod
    def reinStateErrorNewTerm():
        return 60
    @staticmethod
    def bk13FilingMonthsAhead():
        return 3
    @staticmethod
    def bk13HoldingMonthsAhead():
        return 9
    @staticmethod
    def bk13WorkoutPeriod():
        return 60
    @staticmethod
    def bk13LegalCost():
        return 1250
    @staticmethod
    def fcPayoffMonthsAhead():
        return 6
    @staticmethod
    def fcPayoffHaircut():
        return .8
    @staticmethod
    def fcREOMonthsAhead():
        return 6
    @staticmethod
    def fcREOHaircut():
        return .8
    @staticmethod
    def fcREOSalesCommission():
        return .07
    @staticmethod
    def cashFlowMonthsAhead():
        return 3
    @staticmethod
    def modMonthsAhead():
        return 3
    @staticmethod
    def modArrPaymentRatio():
        return 1 # only applies to arrears, then forgive the rest
    @staticmethod
    def yieldMin():
        return .2
    @staticmethod
    def borrowRate():
        return .10
    @staticmethod
    def colHeaders():
        return ['PmtNo','BuyersPrice','SellersPrice','PNL','Rev','Exp',
                'MtgBegBal','MtgPrinPaid','MtgIntPaid','MtgPrinArr','MtgIntArr','MtgEndBal',
                'ArrBegBal','ArrAdded','ArrPaid','ArrForgiven','ArrEndBal',
                'FinCost','DDCost','AcqCost','ServCost','FCCost','RehabCost','BK13Cost','SalesCommission','FirstLienPmt',
                'AmortBegBal','AmortPrinDue','AmortIntDue','AmortEndBal']
    @staticmethod
    def outCols():
        return ['LoanNumber','Address','City','State','ZipCode',
                'CurrIntRate','CurrRemTerm','CurrRemAmt','CurrPmt','CurrRemArr','OrigAmt','OrigStartDate','OrigEndDate',
                'FairMktVal','1stArrBal','1stMtgBal','2ndArrBal','2ndMtgBal','Equity','CLTV+Arr','AcqCLTV+Arr',
                'WAVGPrice','WAVGPrice%',
                'noteSaleNPV','reinStateNPV','bk13NPV','fcPayoffNPV','fcREONPV', 'cashFlowNPV','loanModNPV',
                'noteSaleProb', 'reinStateProb', 'bk13Prob', 'fcPayoffProb', 'fcREOProb', 'cashFlowProb', 'loanModProb']

def read_inputs(inputPath):
# Reads in lien characteristic input file

    dtInput = pd.read_csv('Inputs/'+inputPath,header=0)
    print('\nRead in date template...')

    return dtInput

def amortSchedule(PV,term,r):
# Generates amort schedule given a PV, term, and int rate

    schedule = np.zeros(shape=(term,6))

    n = range(1,term+1)
    pmt = r * PV / (1 - np.power(1+r,-1*term))

    schedule[:,3] = PV * np.power(1+r,n) - pmt *((np.power(1+r,n)-1) / r) # end bal

    schedule[0,0] = PV # start bal
    schedule[1:term,0] = schedule[0:term-1,3]

    schedule[:,1] = schedule[:,0] - schedule[:,3] # prin paid
    schedule[:,2] = schedule[:,0] * r

    schedule[:,4] = schedule[:,1] + schedule[:,2]
    schedule[:,5] = schedule[:,3] / schedule[0,0]

    return pd.DataFrame(data=schedule,index=range(0,term),columns=['BegBal','PrinPaid','IntPaid','EndBal','Pmt','Factor'])

def calculate_ir(loanAmt,term,loanPmt):
# Calculates annual int rate from stream of payments

    pmtArr = np.ones(shape=(term+1)) * loanPmt
    pmtArr[0] = -1 * loanAmt

    return const.monthsInYear()*np.irr(pmtArr)

def initialize_dtCF(dtInput):
# Initializes cash flow pandas datatable.  Adjusts short-termed liens so that they have at least 63 months to accomodate bk13 case

    # adjusts for cases where the bk13 end date (63 months out) exceeds the loan term
    endDate = max(dtt.datetime.strptime(dtInput['2ND MTG MATURITY DATE'],'%m/%d/%y'),
                         dtt.datetime.today().replace(day=1)+du.relativedelta(months=const.bk13FilingMonthsAhead()+const.bk13WorkoutPeriod()))
    datesArr = pd.date_range(start=dtt.date.today().replace(day=1),end=endDate,freq='MS')

    dtCF = pd.DataFrame(data=np.zeros(shape=(datesArr.shape[0],len(const.colHeaders()))),index=datesArr,columns=const.colHeaders())
    dtCF['PmtNo'] = np.arange(1,datesArr.shape[0]+1)

    # switched to calculate IR by using current UPB, not original UPB, which means intRate indicates current int rate and is calculated
    intRate = calculate_ir(dtInput['2ND MTG CURRENT UPB'],
                           pd.date_range(start=dtt.date.today().replace(day=1),end=dtInput['2ND MTG MATURITY DATE'],freq='MS').shape[0],
                           dtInput['2ND MTG CONTRACTUAL MONTHLY PAYMENT'])
    amortArr = amortSchedule(dtInput['2ND MTG CURRENT UPB'],
                             pd.date_range(start=dtt.date.today().replace(day=1),end=dtInput['2ND MTG MATURITY DATE'],freq='MS').shape[0],
                             intRate/const.monthsInYear())
    dtCF.set_value(dtCF.index[amortArr.index],['AmortIntDue','AmortPrinDue','AmortEndBal','AmortBegBal'],
                   amortArr[['IntPaid','PrinPaid','EndBal','BegBal']].values)

    print('Initialized and calculated for mtg: %s ...' %dtInput['LOAN NUMBER'])

    return dtCF, intRate

def adjust_arrears(dtCF,pmtIndx):
# This function adjusts the arrears up and mortgage principal down to incorporate delinquency.

    # Move prin and int due to arrears
    dtCF.set_value(dtCF.index[np.arange(0,pmtIndx,1)],['ArrAdded'],dtCF[['AmortPrinDue','AmortIntDue']].iloc[:pmtIndx].sum(axis=1))
    dtCF.set_value(dtCF.index[np.arange(0,pmtIndx,1)],['MtgPrinArr'],dtCF['AmortPrinDue'].iloc[:pmtIndx])
    dtCF.set_value(dtCF.index[np.arange(0,pmtIndx,1)],['MtgIntArr'],dtCF['AmortIntDue'].iloc[:pmtIndx])

    return dtCF

def loss_mitigation_cost(dtCF,exitStr,loanNum):
# Initializes the due diligence, acquisition, servicing, and foreclosure / bk13 costs in the CF pandas dataframe.

    dtCF.set_value(dtCF.index[0],['DDCost','AcqCost','ServCost'],
                   [const.bpoCost() + const.ownEncReportCost() + const.creditReportCost() + const.pacerCost(),
                    const.recordingFee() + const.wireFee(), const.loanBoardingFee() + const.ownTransferNotice()])

    if (exitStr == 'bk13'):
        dtCF.set_value(dtCF.index[0],['FCCost','BK13Cost'],[const.foreclosureProceeding(),const.bk13LegalCost()])
    elif (exitStr == 'fcREO'):
        dtCF.set_value(dtCF.index[0],['FCCost','RehabCost'],[const.foreclosureProceeding(), const.rehabCost()])
    elif (exitStr == 'fcPayoff'):
        dtCF.set_value(dtCF.index[0],['FCCost'],const.foreclosureProceeding())

    print('Calculated loss mitigation cost for loan: %s, exit: %s ...' %(loanNum,exitStr))

    return dtCF

def exit_pricing(dtCF,exitStr,dtInput):
# Generates cash flows for each exit strategy.


#dtInput = input
#dtCF, intRate = initialize_dtCF(dtInput)

    dtCF.set_value(dtCF.index[0],['MtgBegBal','ArrBegBal'],[dtInput['2ND MTG CURRENT UPB'],dtInput['2ND MTG ARREARS']])

    if (exitStr == 'noteSale'):

        # switched from npv to 10% guaranteed rise in price upon sale
        dtCF.set_value(dtCF.index[const.noteSaleMonthsAhead()],'SellersPrice',np.npv(const.yieldMin() / const.monthsInYear(),
                        dtCF[['AmortPrinDue','AmortIntDue']].iloc[const.noteSaleMonthsAhead()+1:].sum(axis=1)))

        dtCF.set_value(dtCF.index[np.arange(0,const.noteSaleMonthsAhead()+1,1)],['MtgPrinPaid','MtgIntPaid','FinCost','ServCost'],
                       [dtCF['AmortPrinDue'].iloc[:const.noteSaleMonthsAhead()+1],dtCF['AmortIntDue'].iloc[:const.noteSaleMonthsAhead()+1],
                        0,const.specialtyServicingFee()])

    elif (exitStr == 'reinState'):

        # Move prin and int due to arrears
        dtCF = adjust_arrears(dtCF,const.reinStateMonthsAhead())

        # Borrower pays arrears reinStateMonthsAhead
        dtCF.set_value(dtCF.index[const.reinStateMonthsAhead()],['ArrPaid'],dtInput['2ND MTG ARREARS'] + dtCF['ArrAdded'].iloc[:const.reinStateMonthsAhead()].sum(axis=0))

        dtCF.set_value(dtCF.index[np.arange(const.reinStateMonthsAhead(),const.reinStateMonthsAhead()+const.noteSaleMonthsAhead()+1,1)],
                                  ['MtgPrinPaid','MtgIntPaid','FinCost','ServCost'],
                                    [dtCF['AmortPrinDue'].iloc[const.reinStateMonthsAhead():const.reinStateMonthsAhead()+const.noteSaleMonthsAhead()+1].values,
                                    dtCF['AmortIntDue'].iloc[const.reinStateMonthsAhead():const.reinStateMonthsAhead() + const.noteSaleMonthsAhead() + 1].values,
                                    0,const.standardServicingFee()])
        dtCF.set_value(dtCF.index[const.reinStateMonthsAhead()+const.noteSaleMonthsAhead()],['SellersPrice'],
                       np.npv(const.yieldMin()/const.monthsInYear(),dtCF[['AmortPrinDue','AmortIntDue']].iloc[const.reinStateMonthsAhead()+const.noteSaleMonthsAhead()+1:].sum(axis=1)))

    elif (exitStr == 'cashFlow'):

        # There should be no arrears for cashFlow, but if there are they are paid immediately
        dtCF.set_value(dtCF.index[const.cashFlowMonthsAhead()],['ArrPaid'],dtInput['2ND MTG ARREARS'])
        dtCF.set_value(dtCF.index[np.arange(0,const.cashFlowMonthsAhead()+const.noteSaleMonthsAhead()+1,1)],
                       ['MtgPrinPaid','MtgIntPaid','FinCost','ServCost'],
                       [dtCF['AmortPrinDue'].iloc[:const.cashFlowMonthsAhead()+const.noteSaleMonthsAhead()+1],
                        dtCF['AmortIntDue'].iloc[:const.cashFlowMonthsAhead() + const.noteSaleMonthsAhead()+1],
                        0,const.standardServicingFee()])

        dtCF.set_value(dtCF.index[const.cashFlowMonthsAhead() + const.noteSaleMonthsAhead()],['SellersPrice'],
            np.npv(const.yieldMin() / const.monthsInYear(), dtCF[['AmortPrinDue','AmortIntDue']].iloc[const.cashFlowMonthsAhead()+const.noteSaleMonthsAhead()+1:].sum(axis=1)))

    elif (exitStr == 'bk13'):

        # Move prin and int due to arrears
        dtCF = adjust_arrears(dtCF,const.bk13FilingMonthsAhead())

        # Borrower must pay cumulative arrears over a 60 month period
        dtCF.set_value(dtCF.index[np.arange(const.bk13FilingMonthsAhead(),const.bk13FilingMonthsAhead()+const.bk13HoldingMonthsAhead()+1,1)],
                       ['ArrPaid'],(dtInput['2ND MTG ARREARS']+dtCF['ArrAdded'].iloc[:const.bk13FilingMonthsAhead()].sum(axis=0)) / const.bk13WorkoutPeriod())
        dtCF.set_value(dtCF.index[np.arange(const.bk13FilingMonthsAhead()+const.bk13HoldingMonthsAhead()+1,const.bk13FilingMonthsAhead()+const.bk13WorkoutPeriod(),1)],
                       ['AmortPrinDue'],dtCF['AmortPrinDue'].iloc[const.bk13FilingMonthsAhead()+const.bk13HoldingMonthsAhead()+1:const.bk13FilingMonthsAhead()+const.bk13WorkoutPeriod()]
                       +(dtInput['2ND MTG ARREARS'] + dtCF['ArrAdded'].iloc[:const.bk13FilingMonthsAhead()].sum(axis=0)) / const.bk13WorkoutPeriod())
        dtCF.set_value(dtCF.index[np.arange(const.bk13FilingMonthsAhead(),const.bk13FilingMonthsAhead()+const.bk13HoldingMonthsAhead()+1,1)],
                       ['MtgPrinPaid'],dtCF['AmortPrinDue'].iloc[const.bk13FilingMonthsAhead():const.bk13FilingMonthsAhead()+const.bk13HoldingMonthsAhead()+1])
        dtCF.set_value(dtCF.index[np.arange(const.bk13FilingMonthsAhead(),const.bk13FilingMonthsAhead()+const.bk13HoldingMonthsAhead()+1,1)],
                       ['MtgIntPaid'],dtCF['AmortIntDue'].iloc[const.bk13FilingMonthsAhead():const.bk13FilingMonthsAhead()+const.bk13HoldingMonthsAhead()+1])
        dtCF.set_value(dtCF.index[np.arange(0,const.bk13FilingMonthsAhead()+const.bk13HoldingMonthsAhead()+1,1)],['FinCost','ServCost'],
                       [0,const.specialtyServicingFee()])
        dtCF.set_value(dtCF.index[const.bk13FilingMonthsAhead()+const.bk13HoldingMonthsAhead()],['SellersPrice'],
                       np.npv(const.yieldMin()/const.monthsInYear(),dtCF[['AmortPrinDue','AmortIntDue']].iloc[const.bk13FilingMonthsAhead()+const.bk13HoldingMonthsAhead()+1:].sum(axis=1)))

    elif (exitStr == 'fcPayoff'):

        # Move prin and int due to arrears
        dtCF = adjust_arrears(dtCF,const.fcPayoffMonthsAhead())

        # Waterfall of payments
        dtCF.set_value(dtCF.index[const.fcPayoffMonthsAhead()],['ArrPaid','MtgPrinPaid'],
                       [np.maximum(np.minimum(dtInput['2ND MTG ARREARS']+dtCF['ArrAdded'].iloc[:const.fcPayoffMonthsAhead()].sum(),
                                              dtInput['FAIR MARKET VALUE'] * const.fcPayoffHaircut() -
                                              dtInput[['1ST LIEN CURRENT UPB','1ST MTG ARREARS']].sum()),0),
                        np.maximum(np.minimum(dtInput['2ND MTG CURRENT UPB'],
                                              dtInput['FAIR MARKET VALUE'] * const.fcPayoffHaircut() -
                                              dtInput[['1ST LIEN CURRENT UPB', '1ST MTG ARREARS']].sum() -
                                              dtCF['ArrPaid'].iloc[const.fcPayoffMonthsAhead()]), 0)])

        dtCF.set_value(dtCF.index[np.arange(0,const.fcPayoffMonthsAhead()+1,1)],['FinCost','ServCost'],
                       [0,const.specialtyServicingFee()])

    elif (exitStr == 'fcREO'):

        # Move prin and int due to arrears
        dtCF = adjust_arrears(dtCF,const.fcREOMonthsAhead())

        # Waterfall of payments
        totalEquity = dtInput['FAIR MARKET VALUE'] * const.fcREOHaircut() - dtInput[['1ST LIEN CURRENT UPB','1ST MTG ARREARS']].sum()
        dtCF.set_value(dtCF.index[const.fcREOMonthsAhead()],['FirstLienPmt','ArrPaid','MtgPrinPaid','SellersPrice','SalesCommission'],
                       [np.minimum(dtInput['FAIR MARKET VALUE']*const.fcREOHaircut(),dtInput[['1ST LIEN CURRENT UPB','1ST MTG ARREARS']].sum()),
                        np.maximum(np.minimum(totalEquity, dtInput['2ND MTG ARREARS'] + dtCF['ArrAdded'].iloc[:const.fcREOMonthsAhead()].sum()),0),
                        np.maximum(np.minimum(totalEquity - dtCF['ArrPaid'].iloc[const.fcREOMonthsAhead()],dtInput['2ND MTG CURRENT UPB']), 0),
                        np.maximum(totalEquity - dtCF[['ArrPaid', 'MtgPrinPaid']].iloc[const.fcREOMonthsAhead()].sum(),0),
                        dtInput['FAIR MARKET VALUE'] * const.fcREOHaircut() * const.fcREOSalesCommission()])
        dtCF.set_value(dtCF.index[np.arange(0,const.fcREOMonthsAhead()+1,1)],['FinCost','ServCost'],
                       [0,const.specialtyServicingFee()])

    elif (exitStr == 'loanMod'):

        # Move prin and int due to arrears
        dtCF = adjust_arrears(dtCF, const.modMonthsAhead())

        # Forgive arrears in exchange for a large down payment, dont forgive unless they pay down
        dtCF.set_value(dtCF.index[const.modMonthsAhead()],['ArrForgiven','ArrPaid'],
                       [(dtInput['2ND MTG ARREARS'] + dtCF['ArrAdded'].iloc[:const.modMonthsAhead()].sum()) * (1-const.modArrPaymentRatio()),
                        (dtInput['2ND MTG ARREARS'] + dtCF['ArrAdded'].iloc[:const.modMonthsAhead()].sum()) * const.modArrPaymentRatio()])

        dtCF.set_value(dtCF.index[np.arange(const.modMonthsAhead(),const.modMonthsAhead()+const.noteSaleMonthsAhead()+1,1)],
                       ['MtgPrinPaid','MtgIntPaid'],
                       dtCF[['AmortPrinDue','AmortIntDue']].iloc[const.modMonthsAhead():const.modMonthsAhead()+const.noteSaleMonthsAhead()+1].values)
        dtCF.set_value(dtCF.index[const.modMonthsAhead()+const.noteSaleMonthsAhead()],['SellersPrice'],
                       np.npv(const.yieldMin() / const.monthsInYear(), dtCF[['AmortPrinDue', 'AmortIntDue']].iloc[const.modMonthsAhead() + const.noteSaleMonthsAhead() + 1:].sum(axis=1)))
        dtCF.set_value(dtCF.index[np.arange(0,const.modMonthsAhead()+const.noteSaleMonthsAhead()+1)],['FinCost','ServCost'],
                       [0,const.specialtyServicingFee()])

    # Calculates mortgage and arrears beginning and ending balances
    dtCF['MtgEndBal'] = dtCF['MtgBegBal'].iloc[0] - dtCF[['MtgPrinPaid','MtgPrinArr']].sum(axis=1).cumsum()
    dtCF.set_value(dtCF.index[np.arange(1,dtCF.shape[0])],['MtgBegBal'],dtCF['MtgEndBal'].iloc[:-1].values)
    dtCF['ArrEndBal'] = dtCF['ArrBegBal'].iloc[0] - dtCF[['ArrPaid','ArrForgiven']].sum(axis=1).cumsum() + dtCF['ArrAdded'].cumsum()
    dtCF.set_value(dtCF.index[np.arange(1,dtCF.shape[0])],['ArrBegBal'],dtCF['ArrEndBal'].iloc[:-1].values)
    print('Calculated exit pricing for loan: %s, exit: %s ...' %(dtInput['LOAN NUMBER'],exitStr))

    return dtCF

def calculate_pnl(dtCF,exitStr,loanNum):
# Differences the revenues and expenses to calculate exit strategy PNL's.

    dtCF['Rev'] = dtCF[['SellersPrice', 'MtgPrinPaid', 'MtgIntPaid', 'ArrPaid']].sum(axis=1)
    dtCF['Exp'] = dtCF[['BuyersPrice', 'FinCost', 'DDCost', 'AcqCost', 'ServCost',
                            'FCCost', 'RehabCost', 'BK13Cost', 'SalesCommission']].sum(axis=1)

    dtCF['PNL'] = dtCF['Rev'] - dtCF['Exp']
    print('Calculated pnl for loan: %s, exit: %s ...' % (loanNum, exitStr))
    return dtCF

def calculate_exitprobs(arrears,CLTV,acqCLTV):
# Probabilities for weighting cash flows of each exit strategy.  They change based on individual lien's Acq CLTV, CLTV, and arrears amount.

    # guys above 15-35K arrears and CLTV between .6 to .85 will default
    # [0,0,.75,.25,0,0,0]
    # guys above 35K-60K and CLTV between .6 to .85 will default
    # [0,0,.65 * .5,1 - .65 * .5,0,0]
    # guys above 60K and CLTV between .6 to .85 will default
    # [0,0,.5*.4,1 - .5 * .4,0,0]
    # CLTV > .85, calculate acquisition CLTV < .7
    # [0,0,.1,.45,.45,0,0], half are garbage the other half are good

    if (CLTV > .85) and (acqCLTV <= .7):
        exitProbs = [0,0,.1,.45,.45,0,0]

    elif (.6 <= CLTV < .85) and (arrears <= 15000):
        exitProbs = [1,0,0,0,0,0,0]

    elif (.6 <= CLTV < .85) and (15000 < arrears <= 35000):
        exitProbs = [0,0,.75,.25,0,0,0]

    elif (.6 <= CLTV < .85) and (35000 < arrears <= 60000):
        exitProbs = [0,0,.32,.68,0,0,0]

    elif (.6 <= CLTV < .85) and (arrears > 60000):
        exitProbs = [0,0,.2,.8,0,0,0]

    elif (CLTV < .6) and (arrears <= 15000):
        exitProbs = [0,.85,.15,0,0,0,0]

    elif (CLTV < .6) and (arrears > 15000):
        exitProbs = [0,0,.33,.6,.07,0,0]

    else:
        exitProbs = [0,0,0,0,0,0,0]

    return exitProbs

def calculate_yields(dtLiens):
# Generates output file.

    #dtLiens = read_inputs('dateTemplate.csv')
    dtOut = pd.DataFrame(data=np.ones(shape=(dtLiens.shape[0],len(const.outCols()))),columns=const.outCols()) * np.nan
    dtPNL = pd.DataFrame(data=np.zeros(shape=(const.pnlRowMax(),len(const.exitStr()))),columns=const.exitStr())

    dtOut[['LoanNumber','Address','City','State','ZipCode']] = dtLiens[['LOAN NUMBER','ADDRESS','CITY','STATE','ZIP CODE']]
    dtOut[['CurrRemAmt','CurrPmt','CurrRemArr','OrigAmt','OrigStartDate','OrigEndDate']] = dtLiens[['2ND MTG CURRENT UPB','2ND MTG CONTRACTUAL MONTHLY PAYMENT','2ND MTG ARREARS','2ND MTG ORIGINAL UPB','2ND MTG ORIGINAL FUNDING DATE','2ND MTG MATURITY DATE']]
    dtOut[['FairMktVal','1stArrBal','1stMtgBal','2ndArrBal','2ndMtgBal']] = dtLiens[['FAIR MARKET VALUE','1ST MTG ARREARS','1ST LIEN CURRENT UPB','2ND MTG ARREARS','2ND MTG CURRENT UPB']]
    dtOut['Equity'] = dtOut['FairMktVal'] - dtOut[['1stArrBal','1stMtgBal','2ndArrBal','2ndMtgBal']].sum(axis=1,skipna=True)
    dtOut['CLTV+Arr'] = dtOut[['1stArrBal','1stMtgBal','2ndArrBal','2ndMtgBal']].sum(axis=1,skipna=True) / dtOut['FairMktVal']
    dtOut['AcqCLTV+Arr'] = (const.noteBuyHaircut() * dtOut[['2ndArrBal','2ndMtgBal']].sum(axis=1,skipna=True) + dtOut[['1stArrBal','1stMtgBal']].sum(axis=1,skipna=True)) / dtOut['FairMktVal']

    for i, input in dtLiens.iterrows():
        for e,ex in enumerate(const.exitStr()):
#input = dtLiens.iloc[0]
#ex = 'bk13'
#i=0
            # ensure the loan is still active
            if (dtt.datetime.strptime(input['2ND MTG MATURITY DATE'],'%m/%d/%y') > dtt.datetime.today()):

                dtCF, intRate = initialize_dtCF(input)
                dtOut.set_value(i,['CurrIntRate','CurrRemTerm'],[intRate,dtCF.shape[0]])
                dtCF = exit_pricing(dtCF, ex, input)

                dtCF = loss_mitigation_cost(dtCF,ex,input['LOAN NUMBER'])
                dtCF = calculate_pnl(dtCF,ex,input['LOAN NUMBER'])
                dtPNL.set_value(np.arange(0,dtCF.shape[0],1),ex,dtCF['PNL'].values)


                # output a sample of the CF files to check
                if (i==3):
                    dtCF.to_csv('Outputs/'+ex+'.csv')
                    dtPNL.to_csv('Outputs/PNL.csv')

        exitProbs = calculate_exitprobs(dtOut['2ndArrBal'].iloc[i],dtOut['CLTV+Arr'].iloc[i],dtOut['AcqCLTV+Arr'].iloc[i])
        meanCashFlows = np.matmul(exitProbs, dtPNL.transpose().values)

        dtOut.set_value(i,'WAVGPrice',np.npv(const.yieldMin() / const.monthsInYear(), meanCashFlows))
        dtOut.set_value(i,'WAVGPrice%',np.npv(const.yieldMin() / const.monthsInYear(), meanCashFlows) / input[['2ND MTG CURRENT UPB','2ND MTG ARREARS']].sum())

        dtOut.set_value(i,['noteSaleProb', 'reinStateProb', 'bk13Prob', 'fcPayoffProb', 'fcREOProb', 'cashFlowProb', 'loanModProb'],exitProbs)
        dtOut.set_value(i,['noteSaleNPV','reinStateNPV','bk13NPV','fcPayoffNPV','fcREONPV', 'cashFlowNPV','loanModNPV'],
                        [np.npv(const.yieldMin() / const.monthsInYear(), dtPNL[exit]) for exit in const.exitStr()])

    dtOut.to_csv('Outputs/dtOut.csv')
    return dtOut

def main(argv=sys.argv):

    dtLiens = read_inputs('testInput20171227.csv')
    dtOut = calculate_yields(dtLiens)

if __name__ == "__main__":
    sys.exit(main())