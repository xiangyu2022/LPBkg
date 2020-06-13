import numpy as np
from sympy import lambdify, diff
from sympy.abc import x
from scipy import stats
from scipy import integrate
from scipy.optimize import brentq as root
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.interpolate import interp1d
import seaborn as sns


'''
slegendre_polynomials:

Description:
Return the normalized shifted Legendre polynomials according to the given value.

Arguments:

m	
The size of the basis to be considered.

Value:
Numerical values of the m+1 normalized shifted Legendre polynomials.
'''
def slegendre_polynomials(m):
    if m < 0:
        raise ValueError("n must be nonnegative.")
    if m > 20:
        raise ValueError("n must be smaller than 20 to make derivative accurate. Suggest you use rpy2 or R directly to get the correct polynomials.")
    elif m == 0:
        return 1 
    elif m == 1:
        return -1.73205080756888 + 3.46410161513775*x
    elif m == 2:
        return 2.23606797749979 - 13.4164078649987*x + 13.4164078649987*x**2
    elif m == 3:
        return -2.64575131106459 + 31.7490157327751*x - 79.3725393319377*x**2 +  52.9150262212918*x**3 
    elif m == 4:
        return 3 - 60*x + 270*x**2 - 420*x**3 + 210*x**4 
    elif m == 5:
        return -3.3166247903554 + 99.498743710662*x - 696.491205974634*x**2 +  1857.30988259902*x**3 - 2089.4736179239*x**4 + 835.789447169561*x**5 
    elif m == 6:
        return 3.60555127546399 - 151.433153569488*x + 1514.33153569488*x**2 -  6057.3261427795*x**3 + 11357.4865177116*x**4 -9994.58813558618*x**5 +  3331.52937852873*x**6 
    elif m == 7:
        return -3.87298334620742 + 216.887067387615*x - 2927.97540973281*x**2 +  16266.5300540712*x**3 - 44732.9576486957*x**4 + 64415.4590141217*x**5 -  46522.2759546435*x**6 + 13292.0788441839*x**7 
    elif m == 8:
        return 4.12310562561766 - 296.863605044472*x + 5195.11308827825*x**2 -  38097.4959807072*x**3 + 142865.609927652*x**4 - 297160.468649516*x**5 +  346687.213424435*x**6 - 212257.477606797*x**7 + 53064.3694016993*x**8 
    elif m == 9:
        return -4.35889894354067 + 392.300904918661*x - 8630.61990821054*x**2 +  80552.4524766317*x**3 - 392693.205823579*x**4 + 1099540.97630602*x**5 -  1832568.29384337*x**6 + 1795168.94090779*x**7 - 953683.499857264*x**8 +  211929.666634947*x**9 
    elif m == 10:
        return 4.58257569495584 - 504.083326445143*x + 13610.2498140189*x**2 -  157273.997850884*x**3 + 963303.236836668*x**4 - 3467891.652612*x**5 +  7706425.89469334*x**6 - 10694631.8538601*x**7 + 9023595.62669449*x**8 -  4233291.77548631*x**9 + 846658.355097261*x**10 
    elif m == 11:
        return -4.79583152331272 + 633.049761077279*x - 20574.1172350116*x**2 +  288037.641290162*x**3 - 2160282.30967622*x**4 + 9678064.74734944*x**5 -  27421183.4508234*x**6 + 50365438.9913083*x**7 - 59808958.8021786*x**8 +  44302932.4460582*x**9 - 18607231.6273445*x**10 + 3383133.02315354*x**11 
    elif m == 12:
        return 5 - 780.000000000001*x + 30030*x**2 - 500500*x**3 + 4504500*x**4 -  24504480*x**5 + 85765680*x**6 - 199536480*x**7 + 311775750*x**8 -  323323000*x**9 + 213393180*x**10 - 81124680*x**11 + 13520780*x**12 
    elif m == 13:
        return -5.19615242270663 + 945.699740932608*x - 42556.4883419673*x**2 +  832215.772020694*x**3 - 8842292.57771988*x**4 + 57298055.9036248*x**5 -  241925124.926416*x**6 + 691214642.646902*x**7 - 1360828827.71109*x**8 +  1848039148.74345*x**9 - 1700196016.84398*x**10 + 1011686886.05592*x**11  - 351280168.769417*x**12 + 54043102.8876026*x**13 
    elif m == 14:
        return 5.3851648071345 - 1130.88460949825*x + 58805.9996939088*x**2 -  1332935.99306193*x**3 + 16495082.9141414*x**4 - 125362630.147475*x**5 +  626813150.737374*x**6 - 2149073659.671*x**7 + 5171208493.58333*x**8 -  8810207063.14197*x**9 + 10572248475.7704*x**10 - 8737395434.52097*x**11  + 4732755860.36552*x**12 - 1512241517.51324*x**13 +  216034502.501892*x**14 
    elif m == 15:
        return -5.56776436283002 + 1336.26344707921*x - 79507.6751012127*x**2 +    2067199.55263153*x**3 - 29457593.6249993*x**4 + 259226823.899994*x**5 -    1512156472.74996*x**6 + 6110346563.357*x**7 - 17567246369.6514*x**8 +    36435770248.1658*x**9 - 54653655372.2487*x**10 + 58718803292.4986*x**11  - 44039102469.374*x**12 + 21889258032.1149*x**13 - 6477433499.2993*x**14  + 863657799.906573*x**15 
    elif m == 16:
        return 5.74456264653803 - 1562.52103985835*x + 105470.170190438*x**2 -    3117229.4745174*x**3 + 50654978.9609077*x**4 - 510602187.92595*x**5 +    3432381374.39111*x**6 - 16111177879.795*x**7 + 54375225344.3081*x**8 -    134259815664.958*x**9 + 244352864510.224*x**10 - 327150116121.126*x**11  + 318062612895.539*x**12 - 218315166247.826*x**13 +    100246760011.757*x**14 - 27623551647.6841*x**15 + 3452943955.96051*x**16 
    elif m == 17:
        return -5.91607978309961 + 1810.32041362849*x - 137584.351435765*x**2 + 4586145.04785882*x**3 - 84270415.2544058*x**4 + 964053550.510403*x**5 -   7391077220.57976*x**6 + 39821314004.7562*x**7 - 155552007831.079*x**8 +   449372467067.562*x**9 - 970644528865.933*x**10 + 1572283699650.6*x**11 -    1899842803744.48*x**12 + 1686251009240.66*x**13 - 1066811862988.99*x**14  + 455173061541.969*x**15 - 117349304928.789*x**16 +   13805800579.8575*x**17 
    elif m == 18:
        return 6.08276253029822 - 2080.30478536199*x + 176825.906755769*x**2 -   6601500.51888205*x**3 + 136155948.201942*x**4 - 1753688612.84102*x**5 +  15198634644.6221*x**6 - 93052865171.156*x**7 + 415829991233.603*x**8 -    1386099970778.68*x**9 + 3492971926362.27*x**10 - 6697268486909.47*x**11  + 9766849876742.98*x**12 - 10749314065527.8*x**13 +    8774950257573.7*x**14 - 5147970817776.57*x**15 + 2051144622707.85*x**16  - 496817036642.04*x**17 + 55201892960.2267*x**18 
    elif m == 19:
        return -6.24499799839839 + 2373.0992393914*x - 224257.878122486*x**2 +    9319160.71309*x**3 - 214340696.40107*x**4 + 3086506028.17541*x**5 -    30007697496.1498*x**6 + 206991872524.462*x**7 - 1047896354655.09*x**8 +    3984593546095.89*x**9 - 11555321283678.1*x**10 + 25784601211513.1*x**11  - 44406813197605.8*x**12 + 58858734652447.9*x**13 -    59459333985636.2*x**14 + 44924830122480.7*x**15 - 24568266473231.6*x**16  + 9181220688958.53*x**17 - 2096945465996.7*x**18 + 220731101683.863*x**19 
    elif m == 20:
        return 6.40312423743285 - 2689.3121797218*x + 281033.122780928*x**2 -    12927523.6479227*x**3 + 329651853.022028*x**4 - 5274429648.35245*x**5 +    57139654523.8182*x**6 - 440791620612.312*x**7 + 2507002342232.52*x**8 -    10770824877739.7*x**9 + 35543722096541.1*x**10 - 91062428511799.6*x**11  + 182124857023599*x**12 - 284502735232131*x**13 + 345467607067588*x**14  - 322436433263082*x**15 + 226713117138105*x**16 - 116102219157230*x**17  + 40850780814580.9*x**18 - 8826484497333.28*x**19 +   882648449733.327*x**20 


'''
lst_slegendre:

Description:
Return the first m+1 normalized shifted Legendre polynomials according to the given m.

Arguments:

m	
The size of the basis to be considered.

Value:
The first m+1 normalized shifted Legendre polynomials.
'''

def lst_slegendre(m):
    lst = []
    for i in range(m+1):
        lst.append(slegendre_polynomials(i))
    return (lst)
'''
Legj:
Description:
Evaluates the a basis of normalized shifted Legendre polynomials over a specified data vector.

Arguments:
u	
Data vector on which the polynomials are to be evaluated.

m	
The size of the basis to be considered.

Value:
Numerical values of the first m normalized shifted Legendre polynomials.

'''
def Legj(u,m):
    u=np.array(u)
    m = min(len(np.unique(u))-1, m)
    leg_js = lst_slegendre(m)
    a=np.zeros(len(leg_js)*len(u)).reshape(len(leg_js),len(u))
    for i in range(len(leg_js)):
        for j in range(len(u)):
            a[i][j]=lambdify(x,leg_js[i])(u[j])
    return np.transpose(a[1:,:])
'''
denoise:

Description:
Selects the largest coefficients according to the AIC criterion

Arguments:
LP	
Original vector of coefficients estimates. See details.

n	
The dimension of the sample on which the estimates in LP have been obtained.

method	
either “AIC” or “BIC”. See details.

Details:
Give a vector of M coefficient estimates, the largest are selected according to the AIC or BIC criterion as described in Algeri, 2019 and Mukhopadhyay, 2017.

Value:
Selected coefficient estimates.
'''

def denoise(LP, n, method):
    LP = np.array(LP)
    M = len(LP)
    LP2s = np.flip(np.sort(LP**2))
    criterion = np.zeros(len(LP2s))
    if method == "AIC":
        penalty = 2
    if method == "BIC":
        penalty = np.log(n)
    criterion[0] = LP2s[0] - penalty/n
    if criterion[0] < 0:
        return np.repeat(0, M)
    for k in range(M-1):
        criterion[k+1] = np.sum(LP2s[0:k+2]) - (k+2)*penalty/n
    LPsel = LP
    LPsel[LP**2<LP2s[np.argmax(criterion)]] = 0
    return LPsel



'''
BestM

Description:
Computes the deviance p-values considering different sizes of the polynomial basis and selects the one for which the deviance p-value is the smallest.

Arguments:
data	
A vector of data.

g
The postulated model from which we want to assess if deviations occur.

Mmax
The maximum size of the polynomial basis from which a suitable value M is selected (the default is 20). See details.

rg
Range of the data/ search region considered.

Value:
pvals	
The deviance test p-values obtained for each values of M (from 1 to Mmax) considered.

minp	
The minimium value of the deviance p-values observed.

Msel	
The value of M at which the minimum is achieved.
'''

def BestM(data, g, Mmax=20, rg=[-10**7,10**7]):
    data = np.array(data)
    G = lambda c: integrate.quad(g, rg[0], c)[0]
    n = len(data)
    u = np.zeros(n)
    for i in range(n):
        u[i] = G(data[i])
    pval = np.zeros(Mmax)
    S = Legj(u, Mmax)
    LP = np.zeros(len(S[0])) # number of columns
    for i in range(len(S[0])):
        LP[i] = np.sum(S[:,i])/len(S[:,i])
    for m in range(Mmax):
        deviance = n * np.sum(LP[0:m+1]**2)
        pval[m] = stats.chi2.sf(deviance, m+1)
    out = {}
    out["pvals"] = pval
    out["minp"] = np.min(pval)
    out["Msel"] = np.argmin(pval)+1
    return out
'''
c_alpha2:
    
Description
Approximates the quantiles of the supremum of the comparison density estimator using tube formulae and assuming that $H_0$ is true.

Arguments:
M	
The size of the polynomial basis used to estimate the comparison density.

IDs	
The IDs of the polynomial terms to be used out of the M considered.

alpha	
Desired significance level.

c_interval	
Lower and upper bounds for the quantile being computed.

Value:
Approximated quantile of order 1-alpha of the supremum of the comparison density estimator.
'''
def c_alpha2(M, IDs, alpha=0.05, c_interval=[1,10]):
    sleg_f=lst_slegendre(M) 
    Der=sleg_f
    for i in range(len(sleg_f)):
       Der[i]=diff(sleg_f[i],x)
    IDs=[elem +1 for elem in IDs] 
    Derf="0"
    for j in IDs:
        Derf = Derf + "+(" + str(Der[j-1]) + ")**2"
    sqDerf = "(" + Derf +")**(1/2)"
    sqDerf = lambdify(x,eval(sqDerf))
    k0 = integrate.quad(sqDerf, 0, 1)[0]
    whichC = lambda C: 2*(1-stats.norm.cdf(C,0,1))+k0*np.exp(-C**2/2)/np.pi-alpha
    Calp= root(whichC, c_interval[0], c_interval[1])
    return Calp
'''
dhatL2
Description:
Construction of CD-plot and adjusted deviance test. The confidence bands are also adjusted for post-selection inference.

Arguments:
data	
A vector of data. See details.

g	
The postulated model from which we want to assess if deviations occur.

M	
The desired size of the polynomial basis to be used.

Mmax	
the maximum size of the polynomial basis from which M was selected (the default is 20). See details.

smooth	
A logical argument indicating if a denoised solution should be implemented. The default is FALSE, meaning that the full solution should be implemented. See details.

criterion	
If smooth=TRUE, the criterion with respect to which the denoising process should be implemented. The two possibilities are "AIC" or "BIC". See details.

hist.u	
A logical argument indicating if the CD-plot should be displayed or not. The default is TRUE.

breaks	
If hist.u=TRUE, the number of breaks of the CD-plot. The default is 20.

ylim	
If hist.u=TRUE, the range of the y-axis of the CD-plot.

range	
Range of the data/ search region considered.

sigma	
The significance level (in sigmas) with respect to which the confidence bands should be constructed. See details.

Value:
Deviance	
Value of the deviance test statistic.

Dev_pvalue	
Unadjusted p-value of the deviance test.

Dev_adj_pvalue	
Bonferroni adjusted p-value of the deviance test.

kstar	
Number of coefficients selected by the denoising process. If smooth=FALSE, kstar=M.

dhat	
Function corresponding to the estimated comparison density in the u domain.

dhat.x	
Function corresponding to the estimated comparison density in the x domain.

SE	
Function corresponding to the estimated standard errors of the comparison density in the u domain.

LBf1	
Function corresponding to the lower bound of the confidence bands under in u domain.

UBf1	
Function corresponding to the upper bound of the confidence bands in u domain.

f	
Function corresponding to the estimated density of the data.

u	
Vector of values corresponding to the cdf of the model specified in g evaluated at the vector data.

LP	
Estimates of the coefficients.

G	
Cumulative density function of the postulated model specified in the argument g.
'''
def dhatL2(data, g, M=6, Mmax=None, smooth=False, 
           method="AIC", hist_u=True, 
           breaks=20, ylim=[0,2.5], rg=[-10**7,10**7], sigma=2):
    G = lambda y: integrate.quad(g, rg[0], y, limit=200)[0]
    n = len(data)
    u = np.zeros(n)
    for i in range(n):
        u[i] = G(data[i])
    xx = np.arange(rg[0],rg[1],0.001)
    uu = np.zeros(len(xx))
    for i in range(len(xx)):
        uu[i] = G(xx[i])
    S = Legj(u, M)
    LP = np.zeros(len(S[0]),dtype=float) # number of columns
    for i in range(len(S[0])):
        LP[i] = np.sum(S[:,i])/len(S[:,i])
    if smooth is True:
        m = len(LP)
        LP2s = np.flip(np.sort(LP**2))
        criterion = np.zeros(len(LP2s),dtype=float)
        if method == "AIC":
            penalty = 2
        if method == "BIC":
            penalty = np.log(n)
        criterion[0] = LP2s[0] - penalty/n
        if criterion[0] < 0:
            LP=np.repeat(0, m)
        for k in range(m-1):
            criterion[k+1] = np.sum(LP2s[0:k+2]) - (k+2)*penalty/n
        LPsel = LP
        LPsel[LP**2<LP2s[np.argmax(criterion)]] = 0
        LP=LPsel
    IDS0 = np.where(LP==0)
    IDS1 = np.where(LP!=0)
    dhat = interp1d(u, 1+np.matmul(S,LP), fill_value="extrapolate")
    covs=np.matrix(np.cov(S,rowvar=False)*(n-1)/(n**2))
    covs[IDS0, :]=0
    covs[:,IDS0]=0
    vec1 = np.zeros(len(u))
    for j in range(M):
        for k in range(M):
            vec1=vec1+S[:,j]*S[:,k]*covs[j,k]
    SEdhat = interp1d(u, np.sqrt(vec1),  fill_value="extrapolate")
    sigmas0 = np.repeat(1/n,M)
    sigmas0[IDS0] = 0
    SEdhatH0 = interp1d(u, np.sqrt(np.matmul(S**2,sigmas0)) ,fill_value="extrapolate")
    alpha = 1 - stats.norm.cdf(sigma, 0, 1)
    if Mmax>1 and smooth is False:
        alpha = alpha/Mmax
    if Mmax>1 and smooth is True:
        alpha = 2 * alpha/(Mmax + M*(M-1))   
    IDS1 = np.array(IDS1,dtype=np.int32)[0]
    IDS1 = [elem +1 for elem in IDS1]
    qa=c_alpha2(M, IDS1, alpha=alpha, c_interval=[1,20])
    LBf1 = lambda u: 1-qa*SEdhatH0(u)    
    UBf1 = lambda u: 1+qa*SEdhatH0(u)
    SE = lambda u: SEdhat(u)
    deviance = n*np.sum(LP**2)
    pval = stats.chi2.sf(deviance, np.sum(LP!=0))
    adj_pval = None
    if Mmax>1 and smooth is False:
        adj_pval = pval*Mmax
    if smooth is True:
        adj_pval = pval*(Mmax + M*(M-1))/2
    if hist_u is True:
        ones=np.repeat(1,len(uu))
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax=sns.distplot(u,bins=breaks,kde=False,norm_hist=True, ax=ax)
        ax.plot(uu,ones,color='red',linestyle='dashed')
        ax.plot(uu,dhat(uu),color="#1E90FF")
        poly=Polygon(np.stack((np.append(uu,uu[::-1]), np.append(LBf1(uu), UBf1(uu)[::-1])),axis=-1),closed=False,color=(0.1,0,0.03333,0.1))
        ax.add_patch(poly)
        poly2=Polygon(np.stack((np.append(uu,uu[::-1]), np.append(dhat(uu)-SE(uu), (dhat(uu)+SE(uu))[::-1])), axis=-1),closed=False,color=(0, 0, 0.85, 0.17))
        ax.add_patch(poly2)
        ax.set_xbound(lower=min(uu),upper=max(uu))
        ax.set_ybound(lower=ylim[0], upper=ylim[1])
        ax.set_xlabel('G(x)')
        ax.set_ylabel('Comparing Density')
        ax.figure
    dhat_num = lambda x: dhat(G(x))
    dhat_x = lambda x: dhat_num(x)
    f = lambda x: g(x)*dhat(G(x))
    out = {}
    out["Deviance"] = deviance
    out["Dev_pvalue"] = pval
    out["Dev_adj_pvalue"] = adj_pval
    out["dhat"] = dhat
    out["kstar"] = np.sum(np.array(LP)!=0)
    out["dhat_x"] = dhat_x
    out["SE"] = SE
    out["LBf1"] = LBf1
    out["UBf1"] = UBf1
    out["f"] = f
    out["u"] = u
    out["LP"] =LP
    out["G"] = G
    return out

