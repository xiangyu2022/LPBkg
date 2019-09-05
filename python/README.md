# The 'LPBkg' Python package

This package provides a python implemention for the **New Signals Detection Under Background Mismodelling** (LPBkg) algorithm, as proposed by **S. Algeri** in the **"Detecting new signals under background mismodelling", Submitted. arXiv:1906.06615**. LPBkg algorithm provides a unified statistical strategy to perform modelling, estimation, inference, and signal characterization under background mismodelling. The method proposed allows to incorporate the (partial) scientific knowledge available on the background distribution and provides a data-updated version of it in a **purely nonparametric fashion** without requiring the specification of prior distributions.

Further details and theory about the algorithm can be found here [[PDF](https://arxiv.org/pdf/1906.06615.pdf)]

For more technial problems, please contact the package author Xiangyu Zhang at zhan6004@umn.edu.

For more theoretical references, please contact the paper author Sara Algeri at salgeri@umn.edu.

## A quick Setup Guide

### Getting Started 

#### 1. Install the LPBkg package using pip
```bash
python -m pip install LPBkg
```
#### 2. Loading packages and main functions
```bash
from LPBkg.detc import BestM, dhatL2
```

## Tutorial and Examples

Now everything is ready to start our analysis. We consider the Fermi-LAT example described in Section VI of the manuscript *Algeri(2019)*.

The datafiles are available in the folder [[R code and tutorial](https://drive.google.com/file/d/1nikTqVCR-VIxkOL7F6OQAXYlmeoK-AST/view)] and can be loaded as follows:

```bash
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
ca=np.loadtxt('L:\\R code and tutorial\\source_free.txt',dtype=str)[:,1:].astype(float)
bk=np.loadtxt('L:\\R code and tutorial\\background.txt',dtype=str)[:,1:].astype(float)
si=np.loadtxt('L:\\R code and tutorial\\signal.txt',dtype=str)[:,1:].astype(float)
```

To make these data matrix become numpy arrays for further use, we could reshape them as follows:
```bash
cal=ca.reshape(1,len(ca))[0]
bkg=bk.reshape(1,len(bk))[0]
sig=si.reshape(1,len(si))[0]
```

Now, we have stored our source-free sample into the object **cal**, whereas our background and signal physics samples are stored in the objects **bkg** and **sig**, respectively.
We can check the number of observations in each sample and plot their histograms and kernel density estimation as follows:

```bash
len(cal)
len(bkg)
len(sig)
sns.displot(cal, kde=True, norm_hist= True)
sns.displot(bkg, kde=True, norm_hist= True)
sns.displot(sig, kde=True, norm_hist= True)
```


## Background calibration

Now we fit the source free data with a power-law (also known as Pareto type I) over the range [1,35]. This is going to be our postulated background model g_b.

```bash
#This is -loglikelihhod:
def mll(d, y): 
  return -np.sum(np.log(stats.pareto.pdf(y,1,d)/stats.pareto.cdf(35,1,d)))

def powerlaw(y):
    return stats.pareto.pdf(y,1.3590681192057597,scale=1)/stats.pareto.cdf(35,1.3590681192057597,scale=1)

# where the value 1.3590681192057597 is calculated by minimizing function mil with respect to the parameter ''d'' using ''Brent optimization''
```

Let's chek how our postulated model fits the data

```dash
fig, ax = plt.subplots(figsize=(14, 7))
ax=sns.displot(cal, kde=False, norm_hist= True)
uu=np.arange(min(cal),max(cal),0.05)
ax.plot(uu,powerlaw(uu),color='red',linestyle='dashed')
ax.set_xbound(lower=1,upper=15)
ax.set_ybound(lower=0, upper=0.8)
ax.figure
```

In order to assess if g_b is a good model for our background distribution we proceed considering our comparison-density estimator and respective inference. 

## Choice of M

First of all, we can select a suitable value of M (i.e., the number of polynomial terms which will contribute to our estimator) by means of the function bestM, i.e.,
```bash
BestM(data,g, Mmax=20,rg=[-10**7,10**7])
```
The arguments of this function are the following:
- data: the data vector in the original scale. This corresponds to the source-free sample in the background calibration phase and to the physics sample in the signal search phase.
- g: the postulated model from which we want to assess if deviations occur.
- Mmax: the maximum size of the polynomial basis from which a suitable value M is selected (the default is 20).
- rg: range of the data considered. This corresponds to the interval on which the search is conducted.

Let's now see what we obtain when applying this function to our source-free sample, while considering the best our fitted power law as postulated background model.

```dash
BestM(data=cal,g=powerlaw, Mmax=20,rg=[1,35])
```
The largest significance is achieved at M=4 with a p-value of 9.383e-59.

## CD-plots and deviance tests
We can now proceed constructing our CD-plot and deviance test by means of the function dhatL2 below
```dash
dhatL2(data,g, M=6, Mmax=1,smooth=FALSE,criterion="AIC",
       hist.u=TRUE,breaks=20,ylim=[0, 2.5],rg=[-10**7, 10**7],sigma=2)
```

The arguments of this function are the following:
- data: the data vector in the original scale. This corresponds to the source-free sample in the background calibration phase and to the physics sample in the signal search phase.
- g: the postulated model from which we want to assess if deviations occur.
- M: the size of the basis for the full solution, i.e., the estimator dhat(u;G,F).
- Mmax: the maximum size of the polynomial basis from which M was selected. The default is **20**.
- smooth: a logical argument indicating if a denoising process has to be implemented. The default is **FALSE**, meaning that the full solution should be implemented.
- criterion: if **smooth=TRUE**, the criterion with respect to which the denoising process should be implemented (the two possibilities are **"AIC" or "BIC"**).
- hist_u: a logical argument indicating if the CD-plot should be displayed or not. The default is **TRUE**.
- breaks: if hist_u is TRUE, the number of breaks of the CD-plot. The default is **20**.
- ylim: if hist_u is TRUE, the range of the y-axis of the CD-plot.
- rg: range of the data considered. This corresponds to the interval on which the search is conducted.
- sigma: the significance level at which the confidence bands should be constructed. Notice that if Mmax>1 or smooth=TRUE, Bonferroni's correction is automatically implemented. The default value of sigma is **2**.

Let's now see what we get when applying this function to our source-free sample.  We consider the full solution (i.e., we do not apply any denoising criterion), but we must specify that the selected value Mmax=4 was choosen from a pool of M=20 candidates.
```dash
comp = dhatL2(data=cal,g=powerlaw, M=4, Mmax=20, smooth=FALSE,hist_u=TRUE, breaks=20,ylim=[0,2.5],rg=[1,35],sigma=2)
```

Now let's take a look at the values contained in the **comp.density** object. We can extract the value of the deviance test statistics, its unadjusted and adjusted p-values using the following instructions:
```dash
comp['Deviance']
comp.density['Dev_pvalue']
comp.density['Dev_adj_pvalue']
```
Furthermore, we can create new functions corresponding to the estimated comparison density in both the u and the x scale and plot them in order to understand where the most prominent departures occur.

```dash
# Estimated comparison density in u scale.
fig, ax = plt.subplots(figsize=(14, 7))
u=np.arange(0, 1, 0.001)
dhat=np.zeros(len(u))
for i in range(len(u)):
    dhat[i] = comp['dhat'](u[i])
ax.plot(u,dhat,color='dodgerblue')
ax.set_xbound(lower=0,upper=1)
ax.set_ybound(lower=0.6, upper=1.1)
ax.set_xlabel('u', size=15)
ax.set_ylabel('Comparision density', size=15)
ax.set_title('Comparison density on u−scale', size=20)
ax.figure
```

```dash
# Estimated comparison density in x scale.
fig, ax = plt.subplots(figsize=(14, 7))
u=np.arange(min(cal),max(cal),0.05)
dhat_x= np.zeros(len(u))
for i in range(len(u)):
    dhat_x[i] = comp['dhat_x'](u[i])
ax.plot(u,dhat_x,color='dodgerblue')
ax.set_xbound(lower=0,upper=36)
ax.set_ybound(lower=0.6, upper=1.1)
ax.set_xlabel('x', size=15)
ax.set_ylabel('Comparision density', size=15)
ax.set_title('Comparison density on x−scale', size=20)
ax.figure
```

Similarly, we can define a new function corresponding to the estimate of f_b(x) and see how its fit compares to the histogram of the data.
```dash
fig, ax = plt.subplots(figsize=(14, 7))
fb_hat = comp['f']
ax=sns.distplot(cal,bins=30,kde=False,norm_hist=True)
xx=np.arange(min(cal),max(cal),0.05)
fbhat= np.zeros(len(xx))
for i in range(len(xx)):
    fbhat[i] = fb_hat(xx[i])
ax.set_xbound(lower=1, upper=35)
ax.plot(xx,fbhat,color='dodgerblue')
ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.set_title('Source free sample and calibrated background density')
ax.figure
```
  
There are several other values and function which are generated by the dhatL2 function and are summarized below. To exctract them, it is sufficient to use the ['their name'] symbol.(e.g., comp['f'], comp['u'], etc. ).

- Deviance: value of the deviance test statistic.
- Dev_pvalue: unadjusted p-value of the deviance test. 
- Dev_adj_pvalue: adjusted p-values of the deviance test. If smooth=FALSE, it is computed as in formula (19) in **Algeri  (2019)**. If smooth=TRUE, it is computed as in formula (28) in **Algeri  (2019)**. 
- kstar: number of coefficients selected by the denoising process. If smooth=FALSE returns kstar=M.
- dhat: function corresponding to the estimated comparison density in the u scale.
- dhat_x: function corresponding to the estimated comparison density in the x scale.
- SE: function corresponding to the estimated standard errors of the comparison density in the u scale.
- LBf1: function corresponding to the lower bound of the confidence bands under H_0 in u scale.
- UBf1: function corresponding to the upper bound of the confidence bands under H_0 in u scale.
- f: function corresponding to the estimated density of the data and obtained as in equation (10)
in **Algeri  (2019)**.
- u: values u_i=G(x_i),with i=1,...,n, on which the comparison density has been estimated.
- LP: estimates of the LP_j coefficients. If smooth=TRUE, non-zero values corresponds to the k^M estimates in the denoised solution d^(u;G,F)
- G: cumulative density function of the postulated model specified in the argument g.


## Signal search

We can assess if our physics sample provides evidence in favour of the signal using again the functions bestM and dhatL2. 

Below we work on the signal sample and we compare its distribution with the background distribution calibrated as describe in the previous section and which we called fb_hat. This is the equivalent of f^_b(x) in (14) of **Algeri (2019)**.  

```dash
fb_hat=comp['f']
fig, ax = plt.subplots(figsize=(14, 7))
ax=sns.distplot(cal,bins=30,kde=False,norm_hist=True)
xx=np.arange(min(sig),max(sig),0.05)
fbhat= np.zeros(len(xx))
for i in range(len(xx)):
    fbhat[i] = fb_hat(xx[i])
ax.set_xbound(lower=1, upper=35)
ax.plot(xx,fbhat,color='mediumpurple')
ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.set_title('Physics signal sample and calibrated background density')
ax.figure
```

We select the value M which leads to the strongest significance. Notice that now we must specify fb_hat in the argument g.
```dash
BestM(data=sig,g=fb_hat, Mmax=20, rg=[1,35])
```
The selection process based on the deviance test suggests M=3, which we now use to to estimate the comparison density using the dhatL2 function.

```dash
comp_sig=dhatL2(data=sig,g=fb_hat, M=3, Mmax=20, smooth=False,hist_u=True,
                         breaks=20,ylim=[0,2.5],rg=[1,35],sigma=2)
adjusted_pvalue=comp_sig['Dev_adj_pvalue']
#To convert the p-value in terms of sigma significance
sigma_significance=np.abs(stats.norm.ppf(adjusted_pvalue, 0, 1))
adjusted_pvalue
sigma_significance
```
The CD-plot and the deviance test suggest that a signal is present in the region [2, 3.5] with 3.317 sigma significance.

We can further explore the comparison density by plotting it on the $x$-domain and focusing on the [1,5] region. 
```dash
fig, ax = plt.subplots(figsize=(14, 7))
u=np.arange(min(sig),max(sig),0.05)
dhat_x= np.zeros(len(u))
for i in range(len(u)):
    dhat_x[i] = comp_sig['dhat_x'](u[i])
ax.plot(u,dhat_x,color='dodgerblue')
ax.set_xbound(lower=1,upper=5)
ax.set_ybound(lower=0.6, upper=1.6)
ax.set_xlabel('x', size=15)
ax.set_ylabel('d^(G(x), G, F)', size=15)
ax.set_title('Comparison density on X-scale', size=20)
ax.figure
```

It seems like the signal is concentrated between 2 and 3.

Furthermore, we can try repeating the same analysis with a larger basis, say M=6. Here we can set Mmax=1 since we are not doing any model selection, we are just picking M=6.
```dash
comp_sig2=dhatL2(data=sig,g=fb_hat, M=6, Mmax=1, smooth=False,hist_u=True,
                     breaks=20,ylim=[0,2.5],rg=[1,35],sigma=2)
comp_sig2['Dev_adj_pvalue']
pvalue2=comp_sig2['Dev_pvalue']
sigma_significance2=np.abs(stats.norm.ppf(pvalue2, 0, 1))
pvalue2
sigma_significance2

```
Since no selection process was considered, no adjusted p-value is returned and the unandjusted p-value leads to a
3.745 sigma significance. This is larger than the one we obtained before, but we somehow cheated as we have ignored the selection process!
Notice that, that despite no correction was applied the confidence bands are much larger than before but the estimated comparison density is somehow more concentrated around u=0.7. That is simply because a larger basis leads to a reduction of the bias and an increment of the variance. But what do we get if we implement  a denoising process?
To do so we only need to specify smooth=TRUE and select a denoising criterion between AIC or BIC. Just for the sake of consistency with  **Algeri (2019)**, we choose the AIC criterion.

```dash
comp_sig3=dhatL2(data=sig,g=fb_hat, M=6, Mmax=1, smooth=True,
                     method="AIC",hist_u=True,breaks=20,
                     ylim=[0,2.5],rg=[1,35],sigma=2)
comp_sig3['kstar']
adjusted_pvalue3=comp_sig3['Dev_adj_pvalue']
sigma_significance3=np.abs(stats.norm.ppf(adjusted_pvalue3, 0, 1))
adjusted_pvalue3
sigma_significance3
```
By definition the denoising process implies that a selction has been made so we do have an adjusted p-value.
Notice that out of the initial M=6 estimates only kstar_6=4 for of them contribute to the estimator d^\*(u;G,F) plotted above. The estimate of the comparison density does not show any substantial difference compared with the full solution
d^(u;G,F). However, the significance of the deviance test has reduced (3.520sigma).

## References

Algeri S. (2019). Detecting new signals under background mismodelling. **Submitted.** arXiv:1906.06615

## License

The software is subjected to the GNU GPLv3 licensing terms and agreements.
