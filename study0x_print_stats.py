def print_stats(df):
    for p in ['pred_mean', 'pred_median', 'pred_95p', 'pred_max']:
        print()
        print('Prediction using %s'%p)
        print('Mean Error: %0.2f hours'%((df.ttc - df[p])/60/60).mean())
        print('Std Error: %0.2f hours'%((df.ttc - df[p])/60/60).std())
        percentiles = [.05,.25,.50,.75,.95,1.00]
        for perc in percentiles:
            print('Percentage of predictions with less than %0.2f%% relative error: %0.2f%%'%(perc*100, (abs((df.ttc - df[p])/df.ttc) < perc).astype(int).sum()/len(df)*100))
        percentiles = np.percentile(df.ttc, [5,25,50,75,95,100])
        for perc in percentiles:
            print('Percentage of predictions with less than %0.2f min of absolut error: %0.2f%%'%(perc, (abs(df.ttc - df[p]) < perc).astype(int).sum()/len(df)*100))

data= pd.read_csv('data/rule_ttc_predction_study03.csv')
print_stats(data)
data = pd.read_csv('data/rule_ttc_predction_study04.csv')
print_stats(data)
data = pd.read_csv('data/rule_ttc_predction_study05.csv')
print_stats(data)
data = pd.read_csv('data/rule_ttc_predction_study07.csv')
print_stats(data)
