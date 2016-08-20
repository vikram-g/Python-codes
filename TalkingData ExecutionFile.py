# Basic model with events and phone brand alone

from td_mod import *

strt_time = time.time()
# Working directory
wd = "/home/vikram/kaggle/TalkingData"

# Loading the data
df = MakePredictions('GridSearch')
'''
gs_parameters = {
        "objective": "multi:softprob",
        "num_class": 12,
        "booster" : "gbtree",
        "eval_metric": "mlogloss",
        "eta": [0.05,0.1],
        "max_depth": [2,4,5],
        "subsample": [0.6,0.75],
        "colsample_bytree": [0.2,0.3],
        "silent": 1,
        "seed": 8,
	    "rounds": [300],
		"gamma":[0],
		"min_child_weight" : [50,100],
		"alpha" : [0],
		"scale_pos_weight": [0]
    }
'''
gs_parameters = {
        "objective": "multi:softprob",
        "num_class": 12,
        "booster" : "gbtree",
        "eval_metric": "mlogloss",
        "eta": [0.1],
        "max_depth": [2],
        "subsample": [0.7],
        "colsample_bytree": [0.4],
        "silent": 0,
        "seed": 8,
	    "rounds": [300],
		"gamma":[0],
		"min_child_weight" : [100],
		"alpha" : [0],
		"scale_pos_weight": [0]
    }

gs = pd.DataFrame([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).transpose()
gs.columns = ['Eta','MaxDepth','Rounds','Col_sample','Row_sample','Gamma','Weight','Alpha','Dev-Logloss','Val-Logloss']
iter = 0

for eta in gs_parameters['eta']:
        for max_depth in gs_parameters['max_depth']:
            for rounds in gs_parameters['rounds']:
                for colsample_bytree in gs_parameters['colsample_bytree']:
                    for subsample in gs_parameters['subsample']:
                        for gamma in gs_parameters['gamma']:
                            for min_child_weight in gs_parameters['min_child_weight']:
                                for alpha in gs_parameters['alpha']:
                                    ROUNDS = rounds
                                    gboost_params = { 
                                    "objective": gs_parameters['objective'],
                                    "scale_pos_weight" : 1 , #np.array([1,2,3,4,5,6,7,8,9,10,11,12]), #gs_parameters['scale_pos_weight'],
                                    "booster": "gbtree",
									"num_class": gs_parameters['num_class'],
                                    "eval_metric": "mlogloss",
                                    "eta": eta, 
                                    'gamma' : gamma,
                                    "subsample" : subsample,
                                    "alpha" : alpha,
                                    "colsample_bytree" : colsample_bytree,
                                    "max_depth" : max_depth,
                                    "min_child_weight" : min_child_weight,
                                    "silent" : 1
                                    }
                                    gs.ix[iter,:8] = [eta, max_depth, rounds,colsample_bytree,subsample,gamma,min_child_weight,alpha]
                                    print "Iteration started #%d" %(iter)
                                    df.fit(early_stopping_rounds = 25, getImportance = False, params = gboost_params)
                                    df.score()
                                    gs.ix[iter,8:] = df.score_train, df.score_test
                                    print "Iteration ended :#%d; LogLoss: %f Time elapsed: %f Time now: %s" %(iter,df.score_test,round((time.time() - strt_time)/60, 2),datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
                                    gs.to_csv('GridSearch_5.csv')
                                    iter = iter + 1


