from sklearn.pipeline import Pipeline,make_pipeline
from itertools import permutations,combinations
from sklearn.linear_model import RidgeCV,Ridge, LinearRegression
#from linear_regresor import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from functions import get_regression_metrics
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
#to avoid SettingWithCopyWarning
pd.options.mode.chained_assignment = None

link = 'link_agnostic'
modelname = 'Model'
##
# PREDICT RTIME, QTIME and NTIME separatedly


#X = np.array(list(zip(q, total_queued_fts, total_bytes, share, sizes, p, populated_links, max_actives)))
observables = {}
observables[0] = 'linkqueued_mean'          # mean queued at link
observables[1] = 'linkbqueued_mean'         # mean queued bytes at link 
observables[2] = 'linkqueued_median'        # median queued at link 
observables[3] = 'linkbqueued_median'       # median queued bytes at link
observables[4] = 'linkqueued_95p'           # 95 percentile queued at link
observables[5] = 'linkbqueued_95p'          # 95 percentile queued bytes at link
observables[6] = 'linkqueued_max'           # max queued at link
observables[7] = 'linkbqueued_max'          # max queued bytes at link
observables[8] = 'linksactive_max'          # number of active links at submission
observables[9] = 'linkmeanthr'              # mean thr of the link
observables[10] = 'linkmeanqtime'           # average queue time of the link
observables[11] = 'retry_count'           # average queue time of the link
observables[12] = 'SIZE'           # average queue time of the link
observables[13] = 'ftsqueued_mean'          # mean queued at link
observables[14] = 'ftsbqueued_mean'         # mean queued bytes at link 
observables[14] = 'ftsqueued_median'        # median queued at link 
observables[15] = 'ftsbqueued_median'       # median queued bytes at link
observables[16] = 'ftsqueued_95p'           # 95 percentile queued at link

#degree = 3
algo = 'lss'
for degree in [1]:
    for target in ['RTIME', 'QTIME', 'NTIME']:
        r2best = 0
        indexbest = 0
        dfmetrix_train = pd.DataFrame(columns=['explained_variance', 'max_error', 'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'median_absolute_error', 'r2_score', 'model_name'])
        dfmetrix_test = pd.DataFrame(columns=['explained_variance', 'max_error', 'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'median_absolute_error', 'r2_score', 'model_name'])
        for i in range(1):
            print('Training for target %s'%target)
            modelindex = 0
            linkname = 'link-agnostic'
            dname = 'data/trainpolydata_1Million_xfers_2019.csv'
            #dname = 'data/testpolydata_nxfers20_2019.csv'
            models_log = open('models_score.log','w')
            models_score = open('models_score_%s.csv'%(linkname),'w')

            df = pd.read_csv(dname)
            df['stimes'] = pd.to_datetime(df.submitted)
            np.random.seed(42)
            df_train = df.sample(500)
            #df_train['stimes'] = pd.to_datetime(df_train.submitted) #index
            ytr = df_train[target]
            stimes_train = df_train.stimes
            
            #df_test = pd.read_csv(dname).loc[1000000:4000000]
            df_test = df
            #df_test['stimes'] = pd.to_datetime(df_test.submitted) #index
            yte = df_test[target]
            stimes_test = df_test.stimes

            for ncomponents in range(1,7):
                for idx in combinations(list(observables.values()), ncomponents):
                    model_name = 'data/polymodels4/model_%s_polydeg%d_%s_trainedwith_%s_%04d__'%(target, degree,algo,linkname,modelindex)+'-'.join(idx)+'__.model'
                    Xtr = pd.DataFrame(df_train, columns=idx).values
                    #### TRAINING PHASE ###
                    if algo == 'ls':
                        Xtr = df_train[list(idx)]
                        Xtr['bias'] = np.ones(len(Xtr))
                        model = sm.OLS(ytr, Xtr ).fit()
                    if algo == 'lsn':
                        model = LinearRegression(normalize=True, fit_intercept=False)
                    if algo == 'lss':
                        model = LinearRegression()
                    if algo == 'lspca1qlog':
                        model = make_pipeline(PCA(n_components=1), LinearRegression(normalize=True, fit_intercept=False))
                    if algo == 'lspca2':
                        model = make_pipeline(PCA(n_components=2), LinearRegression(normalize=True, fit_intercept=False))
                    if algo == 'lspca3qlog':
                        model = make_pipeline(PCA(n_components=3), LinearRegression())
                    if algo == 'lspca4':
                        model = make_pipeline(PCA(n_components=4), LinearRegression(normalize=True, fit_intercept=False))
                    if algo == 'lspca5':
                        model = make_pipeline(PCA(n_components=5), LinearRegression(normalize=True, fit_intercept=False))
                    if algo == 'ridge':
                        model = make_pipeline(PolynomialFeatures(degree), Ridge(normalize=True))
                    if algo == 'ridgecv':
                        model = make_pipeline(PolynomialFeatures(degree), RidgeCV(normalize=True, cv=10))
                    if algo != 'ls':
                        model.fit(Xtr, ytr)
                    ypredtr = model.predict(Xtr)
                    ypredtr[ypredtr < 0] = 0

                    #### TESTING PHASE ####
                    if algo == 'ls':
                        Xte = df_test[list(idx)]
                        Xte['bias'] = np.ones(len(Xte))
                    else:
                        Xte = pd.DataFrame(df_test, columns=idx).values
                    ypredte = model.predict(Xte)
                    ypredte[ypredte < 0] = 0

                    # Get metrics
                    metrixtr = get_regression_metrics(ytr,ypredtr)
                    metrixtr['model_name'] = model_name
                    metrixte = get_regression_metrics(yte,ypredte)
                    metrixte['model_name'] = model_name
                    dfmetrix_train = dfmetrix_train.append(metrixtr,ignore_index=True)
                    dfmetrix_test = dfmetrix_test.append(metrixte, ignore_index=True)

                    # Save models
                    if metrixte['r2_score'] > r2best:
                        r2best = metrixte['r2_score']
                        indexbest = modelindex
                    if metrixte['r2_score'] < 2:
                        f = open(model_name,'wb')
                        s = pickle.dump(model, f)
                        f.close()
                        #print(metrixte.keys())
                        for k in metrixte:
                            try:
                                print('%0.2f'%(metrixte[k]), end=' ', file=models_log)
                            except TypeError:
                                print('%s'%(metrixte[k]), end=' ', file=models_log)
                        print(file=models_log,flush=True)
                        print('%06d %0.3f'%(modelindex, metrixte['r2_score']),end='\r')
                    else:
                        print('%06d %0.3f'%(modelindex, metrixte['r2_score']),end='\r')
                    modelindex += 1
        print('Best model for %s: %d %0.4f r²'%(target,indexbest, r2best))
        dfmetrix_train.to_csv('data/polymodels4/model_%s_polydeg%d_%s_metrix_train.csv'%(target,degree,algo), index=False)
        dfmetrix_test.to_csv('data/polymodels4/model_%s_polydeg%d_%s_metrix_test.csv'%(target, degree,algo), index=False)
