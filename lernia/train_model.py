"""
train_model:
iterate and tune over different models for prediction and regression
"""

import random, json, datetime, re, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import lernia.train_reshape as t_r
import lernia.train_modelList as modL
import lernia.train_score as t_s
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
import sklearn as sk

class trainMod:
    """
    Routine to iterate over different predictors and regressors, tune and compare models
    Args:
        X: Data set for prediction
        y: reference data

    """
    def __init__(self,X,y):
        """load matrices and split test and train sets"""
        if len(X) != len(y):
            raise ValueError("wrong dimensions: X:%d,y:%d" % (X.shape[0],y.shape[0]) )
        #X = self.factorize(X)
        self.X = X
        self.y = y

    def factorize(self,X):
        """factorize all string fields"""
        return t_r.factorize(X)
            
    def setMatrix(self,X):
        """set a new matrix"""
        X = self.factorize(X)
        self.X = np.array(X).astype(float)
        
    def setScore(self,y):
        """set a new score"""
        self.y = np.array(y).astype(int)

    def getX(self):
        """return data set"""
        return self.X

    def gety(self):
        """return score set"""
        return self.y

    def getModel(self):
        """return trained model"""
        if not hasattr(self,"model"):
            print("first train the models .loopMod()")
        return self.model
    
    def modFirst(self,paramF="train.json"):
        """returns the default model"""
        tml = modL.modelList(paramF)
        clf = tml.regL['bagReg']['mod']
        decT = tml.regL['decTree']['mod']
        clf.set_params(base_estimator=decT)
        return clf

    def modPick(self,clf):
        """pick a random configuration set from the grid"""
        tml = modL.modelList()
        pDecT = tml.gridL['decTree']
        pBag = tml.gridL['bagging']
        paraB = clf.get_params()
        del paraB['base_estimator']
        decT = clf.get_params()['base_estimator']
        paraS = decT.get_params()
        k = random.choice(list(pDecT))
        v = random.choice(pDecT[k])
        paraS[k] = v
        k = random.choice(list(pBag))
        v = random.choice(pBag[k])
        paraB[k] = v
        decT.set_params(**paraS)
        clf.set_params(**paraB)
        clf.set_params(base_estimator=decT)
        s = {**paraS, **paraB}
        return clf, s

    def perfReg(self,clf,X_train,X_test,y_train,y_test):
        """a single regressor run"""
        fit_w = clf.fit(X_train,y_train)
        y_pred = fit_w.predict(X_test)
        scor = 2.*np.sqrt( ((y_test-y_pred)**2).sum() )/(y_pred+y_test).sum()
        cor = sp.stats.pearsonr(y_pred,y_test)[0]
        return y_pred, fit_w, {"err":scor,"cor":cor}

    def regMatrix(self,clf,trainL,testL):
        """iterate regressor over location list"""
        predL, scorL, fitL = [], [], []
        for j in range(len(self.X)):
            X = self.X[j]
            y = self.y[j]
            X_train, X_test = X[trainL], X[testL]
            y_train, y_test = y[trainL], y[testL]
            y_pred, fit_w, scor = self.perfReg(clf,X_train,X_test,y_train,y_test)
            scorL.append(scor)
            predL.append(y_pred)
            fitL.append(fit_w)
            print("- subiteration %.2f cor %.2f err %.2f\r" % (j/len(self.X),scor['cor'],scor['err']),end="",flush=True)
        return predL, fitL, scorL

    def runReg(self,trainL,testL,paramF="train.json"):
        """iterate and score over different parameters"""
        clf = self.modFirst(paramF)
        scorS = {}
        t_start = time.perf_counter()
        predL, fitL, scorL = self.regMatrix(clf,trainL,testL)
        scorS['time'] = time.perf_counter()-t_start
        scorS['cor'] = np.mean([x['cor'] for x in scorL if x['cor'] == x['cor']])
        scorS['err'] = np.mean([x['err'] for x in scorL if x['err'] == x['err']])
        self.model = clf
        return predL, fitL, pd.DataFrame(scorS,index=[1])
    
    def tuneReg(self,trainL,testL,nIter=20,paramF="train.json"):
        """iterate and score over different parameters"""
        clf = self.modFirst(paramF)
        scorS = []
        cor = -1.
        fitR = None
        for i in range(nIter):
            t_start = time.perf_counter()
            clf, s = self.modPick(clf)
            predL, fitL, scorL = self.regMatrix(clf,trainL,testL)
            s['time'] = time.perf_counter()-t_start
            s['cor'] = np.mean([x['cor'] for x in scorL if x['cor'] == x['cor']])
            s['err'] = np.mean([x['err'] for x in scorL if x['err'] == x['err']])
            scorS.append(s)
            if s['cor'] > cor:
                self.model = clf
                fitR = fitL
            print("iteration %.2f cor %.2f err %.2f" % (i/nIter,s['cor'],s['err']),end="\n",flush=True)
        return predL, fitL, pd.DataFrame(scorS)
    
    def perfCla(self,clf,trainL,testL):
        """perform a single classification"""
        print(clf['name'],clf['type'],clf['score'])
        t_start = time.perf_counter()
        Nclass = len(np.unique(self.y))
        y = self.y
        if clf['type'] == "class" :
            y = label_binarize(y,classes=np.unique(y))
        elif clf['type'] == "logit" :
            y = y.ravel()
        X_train = self.X[trainL]
        X_test = self.X[testL]
        y_train = y[trainL]
        y_test = y[testL]
        mod = clf['mod'].fit(X_train,y_train)
        y_score = mod.predict_proba(X_test)
        if isinstance(y_score,list):
            scoreL = []
            for i in range(Nclass):
                try:
                    scoreL.append(y_score[i][:,1])
                except:
                    scoreL.append(np.zeros(len(X_test)))
                y_score = np.hstack(scoreL)
            y_score1 = y_test.ravel()
        else :
            y_score = y_score[:,1]
            #y_score = y_score.ravel()
            #y_score = y_score[:,0]
            y_score1 = label_binarize(y_test,classes=range(Nclass))
            y_score1 = y_score1.ravel()
            #y_score1 = y_score1[:,0]
        #print(y_test.shape,y_score.shape,y_score1.shape)
        x_pr, y_pr, _ = sk.metrics.roc_curve(y_score1,y_score)
        train_score = mod.score(X_train,y_train)
        test_score  = mod.score(X_test ,y_test )
        #cv = cross_validate(mod,self.X,self.y,scoring=['precision_macro','recall_macro'],cv=5,return_train_score=True)
        cv = "off"
        fsc = sk.metrics.f1_score(y_test,mod.predict(X_test),average="weighted")
        acc = sk.metrics.accuracy_score(y_test,mod.predict(X_test))
        y_predict = mod.predict(X_test) == y_test
        auc = sk.metrics.auc(x_pr,y_pr)## = np.trapz(fpr,tpr)
        t_end = time.perf_counter()
        t_diff = t_end - t_start
        return mod, train_score, test_score, t_diff, x_pr, y_pr, auc, fsc, acc, cv
    
    def loopMod(self,paramF="train.json",test_size=0.4):
        """loop over all avaiable models"""
        N = len(self.y)
        shuffleL = random.sample(range(N),N)
        partS = [0,int(N*(1.-test_size)),int(N*(1.)),N]
        trainL = shuffleL[partS[0]:partS[1]]
        testL  = shuffleL[partS[1]:partS[2]]
        #self.X_train,self.X_test,self.y_train,self.y_test = sk.model_selection.train_test_split(self.X, self.y,test_size=test_size,random_state=0)
        trainR = []
        model = []
        rocC = []
        tml = modL.modelList(paramF)
        tml.set_params()
        for index in range(tml.nCat()):
            clf = tml.retCat(index)
            if not clf['active']:
                continue
            # try:
            mod, trainS, testS, t_diff, x_pr, y_pr, auc, fsc, acc, cv = self.perfCla(clf,trainL,testL)
            # except:
            #     print('error: returning model')
            #     return clf['mod'], trainR
            trainR.append([clf['name'],trainS,testS,t_diff,auc,fsc,acc,clf["type"]])
            model.append(mod)
            rocC.append([x_pr,y_pr])
            #print("{m} trained {c} in {f:.2f} s".format(m=modN,c=index,f=t_diff))
        trainR = pd.DataFrame(trainR)
        trainR.columns = ["model","train_score","test_score","time","auc","fsc","acc","type"]
        trainR.loc[:,'perf'] = trainR['acc']*trainR['auc']
        trainR = trainR.sort_values(['perf'],ascending=False)
        mod = model[trainR.index.values[0]]
        self.rocC = rocC
        self.trainR = trainR
        y_pred = mod.predict(self.X)
        try:
            y_class = y_pred.dot(range(y_pred.shape[1]))
        except IndexError:
            y_class = y_pred
        self.y_pred = y_pred
        return mod, trainR#, self.y, y_class
 
    def plotRoc(self):
        """plot roc curve"""
        if not hasattr(self,"rocC"):
            print("first train the models .loopMod()")
            return 
        plt.clf()
        plt.plot([0, 1],[0, 1],'k--',label="model          | auc  f1   acc")
        for idx, mod in self.trainR.iterrows():
            plt.plot(self.rocC[idx][0],self.rocC[idx][1],label='%s | %0.2f %0.2f %0.2f ' %
                     (mod['model'],mod['auc'],mod['fsc'],mod['acc']))
            
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right",prop={'size':12})#,'family':'monospace'})
        #plt.savefig(baseDir+'/fig/modelPerformaces.jpg')
        plt.show()

    def save(self,mod,fName):
        """save the model"""
        joblib.dump(mod,fName) 

    def load(self,fName):
        """load a model"""
        clf = joblib.load(fName)
        return clf

    def tune(self,paramF="train.json",tuneF="train_tune.json"):
        """tune all avaiable models"""
        tml = modL.modelList(paramF)
        params = tml.get_params()
        with open(tuneF) as f:
            pgrid = json.load(f)
        for idx in range(len(pgrid)):
            if not pgrid[idx]['active']:
                continue
            print("tuning: " + pgrid[idx]['name'])
            clf = tml.retCat(idx)['mod']
            CV_rfc = GridSearchCV(estimator=clf,param_grid=pgrid[idx]['param_grid'],cv=5,return_train_score=False)
            CV_rfc.fit(self.X, self.y)
            for k,v in CV_rfc.best_params_.items():
                params[idx][k] = v

        with open(paramF,'w') as f:
            f.write(json.dumps(params))

class regName(trainMod):
    """
    overload class trainMod picking one model from train_modelList and its tuning grid
    """
    def __init__(self,X,y,modName="lasso"):
        """call trainKeras constructor"""
        trainMod.__init__(self,X,y)
        self.modName = modName

    def modFirst(self,paramF):
        """returns the default model"""
        tml = modL.modelList(paramF)
        clf = tml.regL[self.modName]['mod']
        # if self.modName == "perceptron":
        #     clf.set_params(hidden_layer_sizes=(self.X.shape[2],))
        return clf

    def modPick(self,clf):
        """pick a random configuration set from the grid"""
        tml = modL.modelList()
        pLasso = tml.gridL[self.modName]
        paraB = clf.get_params()
        k = random.choice(list(pLasso))
        v = random.choice(pLasso[k])
        paraB[k] = v
        clf.set_params(**paraB)
        return clf, paraB

def regressor(X,y,nXval=6,isShuffle=True,paramF="train.json"):
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import BaggingRegressor
        tml = modL.modelList()
        clf = tml.regL['bagReg']['mod']
        decT = tml.regL['decTree']['mod']
        clf.set_params(base_estimator=decT)
        decT = DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=None, splitter='best')
        clf = BaggingRegressor(base_estimator=decT,bootstrap=True, bootstrap_features=False, max_features=1.0,max_samples=1.0, n_estimators=10, n_jobs=1, oob_score=False,random_state=None, verbose=0, warm_start=False)
        N = len(X)
        X = np.array(X)
        X = np.nan_to_num(X)
        y = np.array(y)
        corrL = []
        fitL = []
        if isShuffle:
            shuffleL = random.sample(range(N),N)
        else :
            shuffleL = list(range(N))
        if nXval == 1:
            fit_q = clf.fit(X,y)
            y_pred = fit_q.predict(X)
            return fit_q, {}
        
        for j in range(nXval): #cross validation
            partS = [int(j/nXval*N),int((j+1)/nXval*N)]
            idL = [x for x in range(0,partS[0])] + [x for x in range(partS[1],N)]
            idL = shuffleL[0:partS[0]] + shuffleL[partS[1]:]
            fit_q = clf.fit(X[idL,:],y[idL])
            y_pred = fit_q.predict(X)
            corrL.append(t_s.calcMetrics(y,y_pred))
            fitL.append(fit_q)
            # if np.isnan(corrL)[0]:
            #     return fit_q, [0]        
        if True: # pick a random model
            nRandom = int(nXval*np.random.uniform())
            fit_q = fitL[nRandom]
        else: # pick the best
            fit_q = [fitL[x] for x in range(nXval) if corrL[x] == max(corrL)][0]
        return fit_q, pd.DataFrame(corrL)

def featKnockOut(reg,X,y,shuffle=True,portion=0.8,mode="predict"):
    """feature knock out, recursively remove one feature at time and calculate performances"""
    X1 = X.copy()
    tL = X.columns
    x_val = 16
    perfL = []
    print('all')
    for j in range(x_val):
        y_pred = reg.predict(X1)
        b_pred = 1.*(y_pred > 0.5)
        kpi = t_s.calcMetrics(y,b_pred)
        kpi['feature'] = "all"
        perfL.append(kpi)
    for i in tL:
        print(i)
        X1 = X.copy()
        for j in range(x_val):
            X1.loc[:,i] = np.random.random(X1.shape[0])
            y_pred = reg.predict(X1)
            b_pred = 1.*(y_pred > 0.5)
            kpi = t_s.calcMetrics(y,b_pred)
            kpi['feature'] = "- " + i
            perfL.append(kpi)
    perfL = pd.DataFrame(perfL)
    return perfL
    
def plotFeatImportance(perfL,ax=[None],isCorr=True):
    """boxplot of scores per feature"""
    if ax[0] == None:
        if isCorr: fig, ax = plt.subplots(1,2)
        else:
            fig, ax = plt.subplots(1,1)
            ax = [ax]
        perfL.boxplot(by="feature",column="rel_err",ax=ax[0])
    if isCorr:
        perfL.boxplot(by="feature",column="cor",ax=ax[1])
    for a in ax:
        for tick in a.get_xticklabels():
            tick.set_rotation(15)
    return ax




def saveModel(fit,fName):
    """save a model"""
    joblib.dump(fit,fName)

def loadModel(fName):
    """load a model"""
    return joblib.load(fName)

