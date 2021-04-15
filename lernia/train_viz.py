"""
train_viz:
utils for plotting feature/KPI distribution.
"""

import random, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import geomadi.lib_graph as gra
from sklearn.metrics import confusion_matrix
import albio
import albio.series_stat as s_s
import lernia
import lernia.train_reshape as t_r
import lernia.train_score as t_c
import matplotlib

colorL = ["#5555aaf0", "#8b122870", "#6CAF3070", "#F8B19570", "#F6728070", "#C06C8470", "#6C5B7B70",
          "#355C7D70", "#99B89870", "#2A363B70", "#67E68E70", "#9F53B570", "#3E671470", "#7FA8A370",
          "#6F849470", "#38577770", "#5C527A70", "#E8175D30", "#47474730", "#36363630", "#A7226E30",
          "#EC204930", "#F26B3830", "#F7DB4F30", "#2F959930", "#E1F5C430", "#EDE57430", "#F9D42330",
          "#FC913A30", "#FF4E5030", "#E5FCC230", "#9DE0AD30", "#45ADA830", "#54798030", "#594F4F30",
          "#FE436530", "#FC9D9A30", "#F9CDAD30", "#C8C8A930", "#83AF9B30"]

colorL = ["#5555aa","#8b1228","#6CAF30","#F8B195","#F67280","#C06C84","#6C5B7B","#355C7D","#99B898","#2A363B","#67E68E","#9F53B5","#3E6714","#7FA8A3","#6F8494","#385777","#5C527A","#E8175D","#474747","#363636","#A7226E","#EC2049","#F26B38","#F7DB4F","#2F9599","#E1F5C4","#EDE574","#F9D423","#FC913A","#FF4E50","#E5FCC2","#9DE0AD","#45ADA8","#547980","#594F4F","#FE4365","#FC9D9A","#F9CDAD","#C8C8A9","#83AF9B"]

def plotHist(y,nBin=7,nQuant=0,threshold=2.5,lab="",isLog=False,ax=None,isDense=False):
    """plot histogram and color percentile"""
    if ax == None:
        fig, ax = plt.subplots(1,1)
    y = np.array(y)
    y = y[~np.isnan(y)]
    colors = plt.cm.BuPu(np.linspace(0, 0.5, 10))
    mHist ,xHist = np.histogram(y,bins=(nBin+1))
    y1, psum = t_r.binOutlier(y,nBin=nBin,threshold=threshold)
    if nQuant > 0:
        quantL = [100*(x/nQuant) for x in range(nQuant)] + [100]
        v = np.percentile(y,quantL)
        v[0] = min(y)*.9; v[-1] = max(y)*1.1
        c = pd.cut(psum,v,labels=colorL[:nQuant])
        c[c!=c] = colorL[0]
    else:
        c = colorL[0]
    mHist1 = np.bincount(y1)
    lenx = min(len(psum),len(mHist1))
    psum = psum[:lenx]
    mHist1 = mHist1[:lenx]
    delta = (psum[-1]-psum[1])/float(len(psum))
    if isDense:
        mHist1 = mHist1/sum(mHist1)
    ax.bar(psum,mHist1,width=delta,fill=True,alpha=0.5,color=c,linewidth=2,label=lab)
    #plt.plot(psum[1:],mHist,label=lab,linewidth=2)
    #ax.legend()
    if isLog:
        plt.xscale('log',basex=10)
        plt.yscale('log',basey=10)

def plotDensOverlap(featM,group="variable",yL="value",isOverall=True):
    """Overlapping density plots"""
    cL = colorL
    j = 0
    for i,g in featM.groupby([group]):
        sns.kdeplot(g[yL],gridsize=int(180),color=cL[j],label=i,shade=True)
        j += 1
    if isOverall:
        sns.distplot(featM[yL],hist=True,kde=True,bins=int(180/5),color='#0000BB60',hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4},label="overall")
    plt.legend()

        
def plotDens3d(featL,group="grp",y="y"):
    """plot a 3d histogram after groupby"""
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.collections import PolyCollection
    norm = len(set(featL[group]))
    verts = []
    xs, zs = [], []
    xs = np.arange(0, 1, 0.025)
    j = 0
    for i,g in featL.groupby([group]):
        ys = np.histogram(g[y],len(xs))[0]/g[y].sum()
        ys[0], ys[-1] = 0, 0
        verts.append(list(zip(xs, ys)))
        j += 1
        zs.append(j/norm)
    ys = np.array(ys)
    poly = PolyCollection(verts,facecolor=colorL[:j])
    poly.set_edgecolor('black')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.add_collection3d(poly, zs=zs, zdir='y')
    # ax.set_xlim3d(0, 10)
    # ax.set_ylim3d(0., .3)
    # plt.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.grid(b=None)
    ax.set_zlim3d(0, .3)
        
def featureImportance(X,y,ax=None):
    """display feature importance"""
    if ax == None:
        fig, ax = plt.subplots(1,1)
    import xgboost as xgb
    import operator
    X_train = X
    y_train = y
    dtrain = xgb.DMatrix(x_train, label=y_train)
    gbdt = xgb.train(xgb_params, dtrain, num_rounds)
    importance = gbdt.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.gcf().show()

def plotPairGrid(X):
    """plot a pair grid"""
    def corrfunc(x, y, **kws):
        r, _ = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.annotate("r = {:.2f}".format(r),xy=(.1, .6),xycoords=ax.transAxes,size = 24)

    cmap = sns.cubehelix_palette(light=1, dark = 0.1, hue = 0.5, as_cmap=True)
    sns.set_context(font_scale=2)
    gp = sns.PairGrid(X)
    gp.map_upper(plt.scatter, s=10, color = 'red')
    gp.map_diag(sns.distplot, kde=False, color = 'red')
    gp.map_lower(sns.kdeplot, cmap = cmap)
    #gp.map_lower(corrfunc);
    plt.show()

def plotRelevanceTree(X,y,isPlot=False):
    """feature importance by extra tree classifier"""
    clf = ExtraTreesClassifier()
    clf.fit(X,y)
    featL = clf.feature_importances_
    sortL = sorted(featL)
    sortI = [sortL.index(x) for x in featL]
    if isPlot:
        plt.plot(sorted(featL))
        plt.show()
    return featL, sortL
    
def plotBoxFeat(c_M,y):
    """plot boxplot of features"""
    c_M.loc[:,"y"] = y
    fig, ax = plt.subplots(2,2)
    cL = c_M.columns
    # c_M.boxplot(column="y_dif",by="t_pop_dens",ax=ax[0,0])
    # c_M.boxplot(column="y_dif",by="t_bast",ax=ax[0,1])
    c_M.boxplot(column="y_dif",by=cL[0],ax=ax[0,0])
    c_M.boxplot(column="y_dif",by=cL[1],ax=ax[0,1])
    c_M.boxplot(column="y_dif",by=cL[2],ax=ax[1,0])
    c_M.boxplot(column="y_dif",by=cL[3],ax=ax[1,1])
    ax[0,0].set_ylim(0,3.)
    ax[0,1].set_ylim(0,6.)
    ax[1,0].set_ylim(0,3.)
    ax[1,1].set_ylim(0,6.)
    plt.show()

def plotFeatCorr(t_M,ax=None):
    if ax == None:
        fig, ax = plt.subplots(1,1,figsize=(10,8))
    """plot correlation in feature matrix"""
    corMat = t_M.corr()
    corrs = corMat.sum(axis=0)
    corr_order = corrs.argsort()[::-1]
    corMat = corMat.loc[corr_order.index,corr_order.index]
    ax = sns.heatmap(corMat, vmax=1, square=True,annot=True,cmap='RdYlGn')
    plt.title('Correlation matrix between the features')
    plt.xticks(rotation=15)
    plt.yticks(rotation=45)
    return ax

def plotFeatCorrScatter(t_M):
    """plot correlation + scatter matrix"""
    pd.plotting.scatter_matrix(t_M, diagonal="kde")
    plt.tight_layout()
    plt.xlabel(rotation=15)
    plt.ylabel(rotation=15)
    plt.show()
    
def plotMatrixCorr(X1,X2):
    """plot cross correlation"""
    X1 = np.array(X1)
    X2 = np.array(X2)
    xcorr = np.corrcoef(X1,X2)
    from matplotlib import gridspec
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 3)
    ax1 = fig.add_subplot(gs[:,0])
    ax1.imshow(X1)
    ax2 = fig.add_subplot(gs[:,1])
    ax2.imshow(X2)
    ax3 = fig.add_subplot(gs[0,2])
    ax3.imshow(xcorr)
    ax4 = fig.add_subplot(gs[1,2])
    ax4.hist(xcorr.ravel(), bins=20, range=(0.0, 1.0), fc='k', ec='k')
    gs.update(wspace=0.5, hspace=0.5)
    ax1.set_title("own data")
    ax2.set_title("customer data")
    ax3.set_title("correlation matrix")
    ax4.set_title("correlation overall")
    plt.show()
    #xcorr = np.correlate(X1,X2,mode='full')
    #xcorr = sp.signal.correlate2d(X1,X2)
    #xcorr = sp.signal.fftconvolve(X1, X2,mode="same")

def plotPerfHeat(scorL):
    """ plot performance heatmap"""
    scorP = scorL.copy()
    tL = [x for x in scorL.columns if bool(re.search("r_",x))]
    tP = [x for x in scorP.columns if bool(re.search("v_",x))]
    scorL.sort_values(tL[-1],inplace=True)
    scorP.sort_values(tP[-1],inplace=True)
    sns.set(font_scale=1.2)
    def clampF(x):
        return pd.Series({"perf":len(x[x>0.6])/len(x)})
    scorV = scorL[tL].apply(clampF)
    labL = ["%s-%.0f" % (x,y*100.) for (x,y) in zip(scorV.columns.values,scorV.values[0])]
    scorV = scorP[tP].apply(clampF)
    labP = ["%s-%.0f" % (x,y*100.) for (x,y) in zip(scorV.columns.values,scorV.values[0])]
    cmap = plt.get_cmap("PiYG") #BrBG
    scorL.index = scorL[idField]
    scorP.index = scorP[idField]
    yL = scorL[scorL[tL[-1]]>0.6].index[0]
    yP = scorP[scorP[tP[-1]]>0.6].index[0]
    fig, ax = plt.subplots(1,2)#,sharex=True,sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    ax[0].set_title("train")
    ax[0] = sns.heatmap(scorL[tL],cmap=cmap,linewidths=.0,cbar=None,ax=ax[0])
    ax[0].hlines(y=yL,xmin=tL[0],xmax=tL[-1],color="r",linestyle="dashed")
    ax[0].set_xticklabels(labL)
    ax[1].set_title("validation")
    ax[1] = sns.heatmap(scorP[tP],cmap=cmap,linewidths=.0,cbar=cbar_ax,ax=ax[1])
    ax[1].hlines(y=yP,xmin=tP[0],xmax=tP[-1],color="r",linestyle="dashed")
    ax[1].set_xticklabels(labP)
    for i in range(2):
        for tick in ax[i].get_xticklabels():
            tick.set_rotation(45)
    plt.show()

def plotPerfHeatDouble(scorP,scorL):
    """plot double sided performance heatmap"""
    tL = [x for x in scorL.columns if bool(re.search("r_",x))]
    tP = [x for x in scorP.columns if bool(re.search("r_",x))]    
    scorL.sort_values(tL[-1],inplace=True)
    scorP.sort_values(tP[-1],inplace=True)
    sns.set(font_scale=1.2)
    def clampF(x):
        return pd.Series({"perf":len(x[x>0.6])/len(x)})
    scorV = scorL[tL].apply(clampF)
    labL = ["%s-%.0f" % (x,y*100.) for (x,y) in zip(scorV.columns.values,scorV.values[0])]
    scorV = scorP[tP].apply(clampF)
    labP = ["%s-%.0f" % (x,y*100.) for (x,y) in zip(scorV.columns.values,scorV.values[0])]
    cmap = plt.get_cmap("PiYG") #BrBG
    scorL.index = scorL[idField]
    scorP.index = scorP[idField]
    yL = scorL[scorL[tL[-1]]>0.6].index[0]
    yP = scorP[scorP[tP[-1]]>0.6].index[0]
    fig, ax = plt.subplots(1,2)#,sharex=True,sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    ax[0].set_title("train")
    ax[0] = sns.heatmap(scorL[tL],cmap=cmap,linewidths=.0,cbar=None,ax=ax[0])
    ax[0].hlines(y=yL,xmin=tL[0],xmax=tL[-1],color="r",linestyle="dashed")
    ax[0].set_xticklabels(labL)
    ax[1].set_title("validation")
    ax[1] = sns.heatmap(scorP[tP],cmap=cmap,linewidths=.0,cbar=cbar_ax,ax=ax[1])
    ax[1].hlines(y=yP,xmin=tP[0],xmax=tP[-1],color="r",linestyle="dashed")
    ax[1].set_xticklabels(labP)
    for i in range(2):
        for tick in ax[i].get_xticklabels():
            tick.set_rotation(45)
    plt.show()

def plotPCA(X):
    """calculates and plot a PCA"""
    pca = PCA().fit(X)
    y = np.std(pca.transform(X), axis=0)**2
    x = np.arange(len(y)) + 1
    fig, ax = plt.subplots(1)
    ax.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    ax.set_yscale('log')
    plt.show()
    foo = pca.transform(X)
    bar = pd.DataFrame({"PC1":foo[0,:],"PC2":foo[1,:],"Class":y})
    sns.lmplot("PC1", "PC2", bar, hue="Class", fit_reg=False)
    plt.show()

def plotImg(X,nCol=3):
    """plot images on multiple columns"""
    f, ax = plt.subplots(ncols=nCol)
    N = int(X.shape[0]/(nCol))
    for i,a in enumerate(ax):
        a.imshow(X[i*N:(i+1)*N,:])
    plt.show()

def kpiDis(scor,tLab="",saveF=None,col_cor="y_cor",col_dif="y_dif",col_sum="sum",isRel=True,ax=None):
    """plot cumulative histogram of KPI: standard deviation and correlation"""
    nbin = 20
    nRef = sum(~np.isnan(scor[col_sum]))
    nShare = sum(~np.isnan(scor[col_cor]))
    locShare = [(x/nbin) for x in range(nbin+1)]
    difShare = [np.sum(np.abs(scor[col_dif]) < x)/nRef for x in locShare]
    corShare = [np.sum(scor[col_cor] > x)/nRef for x in locShare]
    cor = scor[col_cor]
    cor = cor[~np.isnan(cor)]
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 4))
    #plt.hist(cor,bins=20,normed=1,histtype='step',cumulative=-1,label='Reversed emp.')
    if isRel:
        ax.plot(locShare,difShare,label="relative error",color="b")
        ax.fill_between(locShare[:9],0,difShare[:9],label="err < 0.4",color="b",alpha=.5)
        ax.fill_between(locShare[:11],0,difShare[:11],label="err < 0.5",color="b",alpha=.3)
    else :
        ax.plot(locShare,difShare,label="relative difference",color="b")
        ax.fill_between(locShare[:5],0,difShare[:5],label="err < 0.2",color="b",alpha=.5)
        ax.fill_between(locShare[:7],0,difShare[:7],label="err < 0.3",color="b",alpha=.3)
    ax.plot(locShare,corShare,label="correlation",color="g")
    ax.fill_between(locShare[12:],0,corShare[12:],label="corr > 0.6",color="g",alpha=.5)
    ax.fill_between(locShare[10:],0,corShare[10:],label="corr > 0.5",color="g",alpha=.3)
    ax.set_ylabel("covered locations")
    ax.set_xlabel("relative value")
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    #fig.patch.set_alpha(0.7)
    ax.patch.set_alpha(0.0)
    ax.grid(b=True, which='major', color='k', linestyle='--',linewidth=0.25,axis="both")
    #ax.minorticks_on()
#    ax.tick_params(color="k")
    ax.set_title(tLab+" - locations: " + str(nShare) + "/" + str(nRef))
    ax.legend()
    if saveF:
        plt.savefig(saveF)
    #    plt.grid()
    else:
        if not ax:
            plt.show()
    return ax

def plotHistogram(y,nbin=20,label="metric",ax=None,isLog=False,bins=40):
    """plot histogram and cumulative distribution"""
    y = np.nan_to_num(y)
    plt.rcParams['axes.facecolor'] = 'white'
    values, base = np.histogram(y,bins=bins)
    values = values/sum(values)
    cumulative = np.cumsum(values)
    values = values/max(values)
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 4))
    if isLog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    # n, bins, patches = ax.hist(y,nbin,normed=1,histtype='step',cumulative=True,label='Empirical')
    # ax.hist(y,bins=bins,normed=1,histtype='step',cumulative=-1,label='Reversed emp.')
    ax.step(base[:-1], cumulative, c='blue',label="empirical",linestyle="-.")
    ax.step(base[:-1], 1.-cumulative, c='green',label="reversed emp",linestyle="-.")
    ax.step(base[:-1], values, c='red',label="histogram",linewidth=2,linestyle="-")
    ax.grid(True)
    ax.legend()
    ax.set_xlabel(label)
    ax.set_ylabel('Likelihood of occurrence')
    ax.grid()
    if not ax:
        plt.show()
    return ax

def plotTimeSeries(g,ax=None,t=[None],mode=""):
    """plot all time series in a data frame on different rows"""
    groups = list(range(g.shape[1]))
    i = 1
    cL = g.columns
    X1 = g.values
    if not ax:
        fig, ax = plt.subplots(1,1)
    if not any(t):
        t = range(g.shape[0])
    for group in groups:
        ax = plt.subplot(len(groups), 1, i)
        if mode == "scatter": ax.scatter(t,X1[:, group],marker="o",alpha=0.03)
        elif mode == "confidence": plotConfidenceInterval(X1[:, group],nInt=5,ax=ax)
        elif mode == "binned": plotBinned(t,X1[:, group],ax=ax,isScatter=False)
        else: ax.plot(t,X1[:, group],linewidth=2)
        ax.set_title(cL[group], y=0.5, loc='right')
        i += 1
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)
    return fig, ax

def plotConfidenceInterval(y,ax=None,label="value",nInt=5,color="blue"):
    """plot a confidence interval and scatter points"""
    x, yd, xf, yf = s_s.serRunAvDev(y,nInt=nInt)
    y_up = np.array(yf) + .5*np.array(yd)
    y_dw = np.array(yf) - .5*np.array(yd)
    if not ax:
        fig, ax = plt.subplots(1,1)
    ax.scatter(x,y,color=color,marker="+",label=label,alpha=.25)
    ax.plot(xf,yf,color=color,linestyle='-.',lw=2,label=label,alpha=.5)
    xf[0] = 0.
    xf[nInt-1] = 1.
    ax.fill_between(xf,y_up,y_dw,alpha=.25,label='std interval',color=color)
    ax.legend()
    return ax

def plotBinned(t,y,ax=None,isScatter=False,color="blue",label="series",alpha=0.25):
    """plot average and confindence interval on bins"""
    df = pd.DataFrame({"y":y,"t":t})
    def clampF(x):
        return pd.Series({"y":np.nanmean(x["y"]),"sy":np.std(x["y"])})
    dg = df.groupby("t").apply(clampF).reset_index()
    if not ax:
        fig, ax = plt.subplots(1,1)
    if isScatter:
        ax.scatter(t,y,color=color,marker="+",label=label,alpha=alpha)
    ax.plot(dg["t"],dg['y'],color=color,linestyle='-.',lw=2,label=label,alpha=.5)
    ax.fill_between(dg["t"],dg['y']-dg['sy']*.5,dg['y']+dg['sy']*.5,alpha=0.15,color=color)
    return ax
    

def plotOccurrence(y,ax=None):
    """plot histogram of occurrences of values"""
    t, n = np.unique(y,return_counts=True)
    if not ax:
        fig, ax = plt.subplots(1,1)
    ax.bar(t,n)
    for tick in ax.get_xticklabels():
            tick.set_rotation(15)
    return ax
    
def plotHeatmap(X,labV=[None],ax=None,vmin=-1,vmax=1,cmap=None):
    """plot a correlation heatmap"""
    if ax == None:
        fig, ax = plt.subplots(1,1)
    if any(labV):
        X = pd.DataFrame(X,columns=labV,index=labV)
    if cmap == None:
        cmap = 'PiYG'
    sns.heatmap(X, vmin=vmin, vmax=vmax, square=True,annot=True,cmap=cmap,ax=ax)
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
    for tick in ax.get_yticklabels():
        tick.set_rotation(15)


def plotCorr(X,labV=[None],ax=None,method="pearson"):
    """plot a correlation heatmap"""
    if ax == None:
        fig, ax = plt.subplots(1,1)
    if any(labV):
        corMat = pd.DataFrame(X,columns=labV).corr(method=method)
    else:
        corMat = np.corrcoef(X.T)
    sns.heatmap(corMat, vmin=-1,vmax=1, square=True,annot=True,cmap='PuOr',ax=ax)
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
    for tick in ax.get_yticklabels():
        tick.set_rotation(15)

        
def plotCrossMatrix(X,ax=[[None]]):
    """produce a plot of four different cross information kpi"""
    from sklearn.metrics.pairwise import chi2_kernel
    from sklearn.svm import SVC
    import sklearn.metrics.pairwise as paired 
    if ax[0][0] == None:
        fig, ax = plt.subplots(2,2)
    def func(x,y):
        return sp.stats.pearsonr(x,y)[0]
    corM = albio.series_stat.cross_funcM(X,func)
    def func(x,y):
        return sp.stats.spearmanr(x,y)[0]
        #return paired.cosine_similarity(x,y)
    ranM = albio.series_stat.cross_funcM(X,func)
    def func(x,y):
        #return sk.feature_selection.mutual_info_classif(x,y)
        return sk.metrics.mutual_info_score(x,y)
        #return lernia.train_metric.mutualInformation(x,y)
    mutM = albio.series_stat.cross_funcM(X,func)
    def func(x,y):
        mean = .5*(np.mean(x)+np.mean(y))
        rmse = np.sqrt((x-y)**2).sum()/len(x)
        err = rmse/mean
        return err
        # return lernia.train_score.relErr(x,y)
    errM = albio.series_stat.cross_funcM(X,func)
    fig, ax = plt.subplots(2,2)

    ax[0][0].set_title("Pearson r")
    ax[0][1].set_title("Spearman r")
    ax[1][0].set_title("Info gain")
    ax[1][1].set_title("rel_err")
    imgM = pd.DataFrame(corM,index=X.columns,columns=X.columns)
    sns.heatmap(imgM, vmax=1, square=True,annot=True,cmap='RdYlGn',ax=ax[0][0])
    imgM = pd.DataFrame(ranM,index=X.columns,columns=X.columns)
    sns.heatmap(imgM, vmax=1, square=True,annot=True,cmap='RdYlGn',ax=ax[0][1])
    imgM = pd.DataFrame(mutM,index=X.columns,columns=X.columns)
    sns.heatmap(imgM, vmax=1, square=True,annot=True,cmap='RdYlGn',ax=ax[1][0])
    imgM = pd.DataFrame(errM,index=X.columns,columns=X.columns)
    sns.heatmap(imgM, vmax=1, square=True,annot=True,cmap='RdYlGn',ax=ax[1][1])
    for a in ax.flat:
        for tick in a.get_xticklabels():
            tick.set_rotation(15)
        for tick in a.get_yticklabels():
            tick.set_rotation(15)

    plt.show()    
    
def plotConfMat(y,y_pred,ax=None):
    """plot confusion matrix"""
    if ax == None:
        fig, ax = plt.subplots(1,1)
    cm = confusion_matrix(y,y_pred)
    cm = np.array(cm)
    plotHeatmap(cm/cm.sum(),ax=ax)
    ax.set_xlabel("prediction")
    ax.set_ylabel("score")
    #plt.grid(b=None)
    return cm, ax

def plotHyperPerf(scorV):
    """plot a double boxplot showing the performances per hyperparameter"""
    scorM = t_r.factorize(scorV.copy())
    setL = ((scorM.var(axis=0)/scorM.mean(axis=0)**2).abs() > 1e-4) | (scorV.columns.isin(['cor','err']))
    scorV = scorV[scorV.columns[setL]]
    scorM = t_r.factorize(scorV.copy())
    tL = [x for x in scorM.columns if not any([x == y for y in ["cor","err","time"]])]
    for i in tL:
        fig, ax = plt.subplots(1,2)
        scorV.boxplot(by=i,column="cor",ax=ax[0])
        scorV.boxplot(by=i,column="err",ax=ax[1])    
        plt.show()
    return scorV

def plotParallel(frame, class_column, cols=None, ax=None, color=None, use_columns=False, xticks=None, colormap=None,**kwds):
    """plot a parallel representation"""
    n = len(frame)
    class_col = frame[class_column]
    class_min = float(np.amin(class_col))
    class_max = float(np.amax(class_col))
    if cols is None:
        df = frame.drop(class_column, axis=1)
    else:
        df = frame[cols]
    used_legends = set([])
    ncols = len(df.columns)
    if use_columns is True:
        if not np.all(np.isreal(list(df.columns))):
            raise ValueError('Columns must be numeric to be used as xticks')
        x = df.columns
    elif xticks is not None:
        if not np.all(np.isreal(xticks)):
            raise ValueError('xticks specified must be numeric')
        elif len(xticks) != ncols:
            raise ValueError('Length of xticks must match number of columns')
        x = xticks
    else:
        x = range(ncols)
    fig = plt.figure()
    ax = plt.gca()
    Colorm = plt.get_cmap(colormap)
    for i in range(n):
        y = df.iloc[i].values
        kls = float(class_col.iat[i])
        ax.plot(x, y, color=Colorm((kls - class_min)/(class_max-class_min)), **kwds)
    for i in x:
        ax.axvline(i, linewidth=1, color='black')
    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.set_xlim(x[0], x[-1])
    ax.legend(loc='upper right')
    ax.grid()
    bounds = np.linspace(class_min,class_max,10)
    cax,_ = mpl.colorbar.make_axes(ax)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=Colorm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%.2f')
    return fig

def plotSankey(df,col_1="1",col_2="2",title="reshuffling"):
    """plot a sankey diagram"""
    from pySankey import sankey
    sankey.sankey(df[col_1],df[col_2],aspect=20,fontsize=12)
    plt.title(title)
    plt.show()

def plotSankeyFlow(df,col_1="1",col_2="2"):
    from matplotlib.sankey import Sankey
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, title = "")
    ax.axis('off')
    df = pd.DataFrame({"flow":[242, -121, -121],"length":[2, 1, 1],"label":["1","2","3"],"orientation":[0, 0, 1]})
    df1 = pd.DataFrame({"flow":[121, -65, -17, -11, -8, -8, -8, -1, -1, -2],"path":[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],"label":["1","2","3","4","5","6","7","8","9","0"],"length":[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],"orientation":[0, 0, 1, -1, 1, -1, 1, -1, 1, -1]    })
    sankey = Sankey(ax=ax,scale=0.02,offset=0.3)
    sankey.add(flows=df['flow'],pathlengths=df['length'],labels=list(df['label']),label='1',orientations=df['orientation'])
    sankey.add(flows=df1['flow'],pathlengths=df1['length'],labels=list(df1['label']),label='2',orientations=df1['orientation'],fc='#FFFF00',prior=0,connect=(1,0))
    diagrams = sankey.finish()
    plt.legend(loc='lower right')
    plt.show()

def plotRadar(featL,idGroup="week",idField="feature",idVal="value",isFill=False):
    """radar plot"""
    feat = featL.sort_values([idGroup,idField])
    categories = list(np.unique(feat[idField]))
    N = len(categories)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    width = np.pi / 4 * np.random.rand(N)
    maxy = max(feat[idVal])
    cmap = plt.get_cmap("Dark2")
    ax = plt.subplot(111, projection='polar')
    plt.xticks(theta[:], categories, color='grey', size=8)
    ax.set_rlabel_position(0)
    j = 0
    if isFill:
        featG = feat.groupby(idField).agg(np.mean)
        meanL = featG[idVal]
        featG = feat.groupby(idField).agg(np.std)
        stdL = featG[idVal]
        meanL = (meanL - 0)/(maxy - 0) 
        stdL = (stdL - 0)/(maxy - 0) 
        ax.fill_between(theta,meanL + stdL,meanL - stdL,alpha=.15,color=colorL[0])
        
    for i,g in feat.groupby(idGroup):
        y = g[idVal]
        y = (y - 0)/(maxy - 0) 
        color = matplotlib.colors.rgb2hex(cmap(j))
        color = colorL[j]
        bars = ax.plot(theta, y,label=i,color=color)
        # ax.plot(theta,y,color=color,alpha=.25)
        j = j + 1
    ax.set_yticks([0.2,0.4,0.6,0.8],[])
    plt.legend(loc='right', bbox_to_anchor=(-.0,.2))

    
def singleRadar(x,y1,ax=None,color="#888888",label=""):
    """single radar plot"""
    if ax == None:
        ax = plt.subplot(111, projection='polar')
    N = len(x)
    x_as = [n / float(N) * 2 * np.pi for n in range(N)]
    y = list(y1) + list(y1[:1])
    x_as += x_as[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_ylim(0,max(y1))
    ax.xaxis.grid(True, color=color, linestyle='solid', linewidth=0.5)
    ax.yaxis.grid(True, color=color, linestyle='solid', linewidth=0.5)
    plt.xticks(x_as[:-1], [])
    ax.plot(x_as, y, linewidth=0, linestyle='solid', zorder=3)
    ax.fill(x_as, y, 'b', alpha=0.3,label=label,color=color)
    ax.set_xticklabels(x,size=14)
    #ax.tick_params(axis='y',which='both',bottom=False,top=False,labelbottom=False)
    ax.set_yticks([],[])
    # ax.fill(theta,X[i],label=labL[i],color=colL[i],linewidth=3,alpha=.4)
    # ax.set_xticklabels(cL,size=7)
    return ax
    
def radarGrid(X,cL,labL=[None]):
    """radar plots in a grid from a matrix"""
    N = X.shape[1]
    N_col = int(np.sqrt(X.shape[0]))
    N_row = int(X.shape[0]/N_col)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    width = np.pi / 4 * np.random.rand(N)
    cmap = plt.get_cmap("Dark2")
    categories = list(cL)
    colL = ['blue','red','green','yellow','purple','brown','olive','cyan','#ffaa44','#441188']
    colL = colL + colL
    if labL[0] == None:
        for i in range(X.shape[0]):
            labL = "cluster"+str(i)
    for i in range(N_col*N_row):
        plt.rc('axes', linewidth=0.5, edgecolor=colL[i])
        ax = plt.subplot(N_col,N_row,i+1,projection='polar')
        singleRadar(cL,X[i],ax=ax,color=colL[i],label=labL[i])
        plt.legend(loc='right', bbox_to_anchor=(0.7, -0.1))
    plt.show()

def plotJoin(vist,col_ref="bast",col1="via",col2="dirc"):
    """produce a join plot"""
    #sns.set(style="white", color_codes=True)
    #sns.jointplot(x=vist['via_r'],y=vist['dirc_r'],kind='hex',s=200,color='m',edgecolor="skyblue",linewidth=2)
    ax0 = sns.jointplot(x=vist[col_ref],y=vist[col1],kind='hex',color='m',edgecolor="skyblue",linewidth=0)
    ax1 = sns.jointplot(x=vist[col_ref],y=vist[col2],kind='hex',color='g',edgecolor="skyblue",linewidth=0)
    fig = plt.figure(figsize=(10,5))
    for J in [ax0, ax1]:
        for A in J.fig.axes:
            fig._axstack.add(fig._make_key(A), A)
    fig.axes[0].set_position([0.2, 0.1, 0.5,  0.6])
    fig.axes[1].set_position([0.2, 0.7, 0.5,  0.1])
    fig.axes[2].set_position([0.7, 0.1, 0.1,  0.6])
    fig.axes[3].set_position([1.0, 0.1, 0.5,  0.6])
    fig.axes[4].set_position([1.0, 0.7, 0.5,  0.1])
    fig.axes[5].set_position([1.5, 0.1, 0.1,  0.6])
    plt.show()

def showPredicted(g,model,X1=np.array([None]),isPlot=False,idField="id_poi"):
    X = g['values']
    X = np.roll(X,shift=1,axis=1)
    rawIm = np.reshape(X, (1,X.shape[0],X.shape[1],1) )
    decIm = model.predict(rawIm)
    X = rawIm[0,:,:,0]
    Y = decIm[0,:,:,0]
    # X = t_r.removeInterp(X,step=3)
    # Y = t_r.removeInterp(Y,step=3)
    X = t_r.removeBackfold(X)
    Y = t_r.removeBackfold(Y)
    x = X.ravel()*g['norm']
    y = Y.ravel()*g['norm']
    r = sp.stats.pearsonr(x,y)[0]
    e = t_c.relErr(x,y)[2]
    ncol = 2
    if X1.all() != None:
        ncol = 3
        X1 = t_r.removeBackfold(X1)
        x1 = X1.ravel()*g['norm']
        r1 = sp.stats.pearsonr(x,x1)[0]
        e1 = t_c.relErr(x,x1)[2]
        r2 = sp.stats.pearsonr(y,x1)[0]
        e2 = t_c.relErr(y,x1)[2]
    if isPlot:
        ax2 = plt.subplot2grid((2, ncol), (0, 0), colspan=ncol)
        ax4 = plt.subplot2grid((2, ncol), (1, 0))
        ax5 = plt.subplot2grid((2, ncol), (1, 1))
        ax2.set_title("id_poi %s week %s corr %.2f rel_err %.2f" % (g[idField],g['week'],r,e) )
        ax2.plot(x,label="reference")
        ax2.plot(y,label="predicted")
        ax4.imshow(X,cmap=plt.get_cmap("viridis"))
        ax4.set_xlabel("reference")
        ax5.imshow(Y,cmap=plt.get_cmap("viridis"))
        ax5.set_xlabel("predicted")
        if X1.all() != None:
            ax2.set_title("id_poi %s week %s corr %.2f->%.2f err %.2f->%.2f" % (g[idField],g['week'],r1,r2,e1,e2) )
            ax2.plot(x1,label="original")
            ax6 = plt.subplot2grid((2, ncol), (1, 2))
            ax6.imshow(X1,cmap=plt.get_cmap("viridis"))
            ax6.set_xlabel("reference")
        ax2.legend()
        plt.show()
    if X1.all() != None:
        return {idField:g[idField],"week":g['week'],"cor_auto":r,"err_auto":e,"cor_ext":r1,"err_ext":e1,"cor_pred":r2,"err_pred":e2}
    return {idField:g[idField],"week":g['week'],"cor":r,"err":e}

def plotPie(df,gL,isValue=True,ax=None):
    """plot a double pie chart (e.g. age&gender)"""
    if ax == None:
        fig, ax = plt.subplots(1,1)
    if isValue:
        z = df[gL].values.ravel()
        h, n1 = t_r.binOutlier(z,nBin=10)
        h, n = np.unique(h,return_counts=True)
        explode = [x*.3 for x in n1[:-1]]
        labels = ["%.2f" % x for x in n1[:-1]]
    else:
        z = df.melt(value_vars=gL)
        col = z.columns[0]
        z = z.groupby(col).agg(np.nansum).reset_index()
        z.loc[:,"value"] = z['value']/sum(z['value'])
        explode = [x*.3 for x in z['value']]
        labels = z[col]
        n = z['value']
    ax.pie(n,explode=tuple(explode),labels=labels,autopct='%1.1f%%',shadow=True,startangle=45)
    return ax

def boxplotOverlap(X1,X2,cL,by=None,lab1='via',lab2='tile',ax = None):
    import matplotlib.patches as mpatches
    if ax == None:
        fig, ax = plt.subplots(1,1)
    if by == None:
        bx1 = X1.boxplot(column=cL,ax=ax,return_type="dict")
        bx2 = X2.boxplot(column=cL,ax=ax,return_type="dict")
        [[item.set_color('blue') for item in bx1[key]] for key in bx1.keys()]
        [[item.set_color('blue') for item in bx1[key]] for key in bx1.keys()]
        [[item.set_color('orange') for item in bx2[key]] for key in bx2.keys()]
        [[item.set_color('orange') for item in bx2[key]] for key in bx2.keys()]
    else:
        bx1 = X1.boxplot(column=cL,by=by,ax=ax,return_type="dict")
        bx2 = X2.boxplot(column=cL,by=by,ax=ax,return_type="dict")
        [[item.set_color('blue') for item in bx1[key]['boxes']] for key in bx1.keys()]
        [[item.set_color('blue') for item in bx1[key]['medians']] for key in bx1.keys()]
        [[item.set_color('orange') for item in bx2[key]['boxes']] for key in bx2.keys()]
        [[item.set_color('orange') for item in bx2[key]['medians']] for key in bx2.keys()]
    blue_patch = mpatches.Patch(color='blue',label=lab1)
    red_patch = mpatches.Patch(color='orange',label=lab2)
    plt.legend(handles=[red_patch, blue_patch])
    plt.xticks(rotation=15)
    plt.show()

def joyplotOverlap(feat,cL,by=None,lab1='via',lab2='tile',ax = None):
    import joypy
    if by == None:
        fig, ax = joypy.joyplot(feat,column=cL,xlim='own',ylim='own',figsize=(12,6),alpha=.5)
    else:
        feat.loc[:,by] = feat.loc[:,by].astype(int)
        feat = feat.melt(id_vars=[by],value_vars=cL)
        sorter = dict(zip(cL, range(len(cL))))
        feat.loc[:,'rank'] = feat['variable'].map(sorter)
        feat = feat.sort_values('rank')
        fL = np.unique(feat[by])
        vL = []
        for i in fL:
            var = by + "_" + str(i)
            feat.loc[:,by+"_"+var] = float('nan')
            setL = feat[by] == i
            feat.loc[setL,var] = feat.loc[setL,"value"]
            vL.append(var)

        fig, ax = joypy.joyplot(feat,column=vL,by="rank",xlim='own',ylim='own',figsize=(12,6),alpha=.5,labels=cL)
    return fig, ax

def plotOccurrence(y,lim=30):
    """bar plot for variable occurrence"""

def heatmapHisto(convM,bins=20):
    """plot an heatmap and the histogram of the x and y axis"""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    x = convM.sum(axis=1)
    y = convM.sum(axis=0)
    fig, axScatter = plt.subplots(figsize=(5.5, 5.5))
    axScatter.imshow(convM)
    #axScatter.set_aspect(1.)
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)
    axHistx.xaxis.set_tick_params(labelbottom=False)
    axHisty.yaxis.set_tick_params(labelleft=False)
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1)*binwidth
    #bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')
    # axHistx.set_yticks([0, 50, 100])
    # axHisty.set_xticks([0, 50, 100])

def scatterRegression(featL,xL,yL):
    """plot scatter points and regression"""
    feat = featL.sort_values(yL)
    for t in xL:
        x, y = feat[yL], feat[t]
        m, b = np.polyfit(x, y, 1)
        plt.scatter(x,y,label=t)
        plt.plot(x, m*x+b)
    plt.legend()

def boxplotMean(featL,x_lab="variable",y_lab="value",ax=None):
    """boxplot and mean"""
    import matplotlib.transforms as transforms
    if ax == None:
        fig, ax = plt.subplots()
    featG = featL.groupby([x_lab]).agg(np.std).reset_index()
    err = featG[y_lab].mean()
    sns.pointplot(x=x_lab,y=y_lab,data=featL,linestyles='',scale=1,color='k',errwidth=err,capsize=0.2,markers='x',ax=ax)
    offset = transforms.ScaledTranslation(5/72., 0, ax.figure.dpi_scale_trans)
    trans = ax.collections[0].get_transform()
    ax.collections[0].set_transform(trans + offset)
    # sns.swarmplot(x=x_lab,y=y_lab,data=featL,edgecolor="black",linewidth=.9,ax=ax)
    sns.boxplot(x=x_lab,y=y_lab, data=featL,saturation=1,ax=ax)
    sns.pointplot(x=x_lab,y=y_lab,data=featL,linestyles='--',scale=0.4,color='k', errwidth=0, capsize=0, ax=ax)
    return ax
