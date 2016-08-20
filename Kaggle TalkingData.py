# Packages
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import time
import datetime
from sklearn.metrics import log_loss
from scipy.sparse import csr_matrix, hstack
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV

# Working directory
wd = "/home/vikram/kaggle/TalkingData"

# Load the data
class TrainTest(object):
    '''
    Loads the data set into train and test data set
    When action =  'GridSearch' train is split into 80-20 sample for cross validation
    When acrion = 'Prediction' then actual train and test are loaded
    '''
    def __init__ (self, action, dv = 'group'):
        self.train = pd.read_csv(os.path.join(wd,'Data', '.'.join(['gender_age_train','csv'])), usecols = ['device_id',dv], index_col = 'device_id')
        if action == 'GridSearch':
            self.train, self.test = train_test_split(self.train, test_size = 0.2)
        elif action == 'Prediction':
            self.test = pd.read_csv(os.path.join(wd,'Data', '.'.join(['gender_age_test','csv'])), usecols = ['device_id'], index_col = 'device_id')
            self.fileIndex = self.test.index
        self.encoder_group = LabelEncoder()
        self.encoder_brand = LabelEncoder()
        self.encoder_model = LabelEncoder()
        self.encoder_app = LabelEncoder()
        self.encoder_label = LabelEncoder()
        self.encoder_dv = LabelEncoder()
        self.action = action
        self.dv = dv
        
    def load_data(self):
        self.train[self.dv] = self.encoder_group.fit_transform(self.train[self.dv])
        if self.action == 'GridSearch':
            self.test[self.dv] = self.encoder_group.transform(self.test[self.dv])
        print 'Train & Test data loaded with Train: %d rows & %d columns and Test: %d rows & %d columns' %(self.train.shape[0],self.train.shape[1],self.test.shape[0],self.test.shape[1])
        self.train['trrow'] = np.arange(self.train.shape[0])
        self.test['terow'] = np.arange(self.test.shape[0])
        # Phone brand data
        self.get_phone_brand()
        
        # Get events data which is passed on to get Apps and labels data at one go
        self.get_app_labels(self.get_events())
        
        # Get the dv
        self.get_dv()
        
        self.train = hstack((self.xtr_brand, self.xtr_model, self.xtr_evnt,self.Xtr_app,self.Xtr_label),format='csr')
        self.test = hstack((self.xte_brand, self.xte_model, self.xte_evnt,self.Xte_app,self.Xte_label),format='csr')
        
        print 'Completed load data'
        
    def get_phone_brand (self):
        # Import phone brand data
        print 'phone brand started'
        pbd = pd.read_csv(os.path.join(wd,'Data', '.'.join(['phone_brand_device_model','csv'])))
        pbd = pbd.drop_duplicates('device_id', keep='first', inplace=False).set_index('device_id')
        # Label encoding the Chinese characters
        pbd['phone_brand'] = self.encoder_brand.fit_transform(pbd['phone_brand'])
        pbd['device_model'] = self.encoder_model.fit_transform(pbd['device_model'])
        nmodels = len(self.encoder_model.classes_)
        self.train['brand'] = pbd['phone_brand']
        self.train['model'] = pbd['device_model']
        
        self.test['brand'] = pbd['phone_brand']
        self.test['model'] = pbd['device_model']
        self.train = self.train.merge(pbd, left_index = True, right_index = True, how = 'left')
        self.test = self.test.merge(pbd, left_index = True, right_index = True, how = 'left')
        self.xtr_brand = csr_matrix ((np.ones(self.train.shape[0]), (self.train.trrow, self.train.brand)))
        self.xte_brand = csr_matrix ((np.ones(self.test.shape[0]), (self.test.terow, self.test.brand)))
        self.xtr_model = csr_matrix ((np.ones(self.train.shape[0]), (self.train.trrow, self.train.model)), shape = (self.train.shape[0],nmodels))
        self.xte_model = csr_matrix ((np.ones(self.test.shape[0]), (self.test.terow, self.test.model)), shape = (self.test.shape[0],nmodels))
        print 'Brand matrix created with Train : %d Rows and %d Cols; Test : %d Rows and %d Cols' %(self.xtr_brand.shape[0],self.xtr_brand.shape[1],self.xte_brand.shape[0],self.xte_brand.shape[1])
        print 'Model matrix created with Train : %d Rows and %d Cols; Test : %d Rows and %d Cols' %(self.xtr_model.shape[0],self.xtr_model.shape[1],self.xte_model.shape[0],self.xte_model.shape[1])
        self.train, self.test = self.train[['trrow',self.dv]],self.test[['terow',self.dv]]

    def get_events(self):
        # Import events data
        events = pd.read_csv(os.path.join(wd,'Data', '.'.join(['events','csv'])))
        print 'Events data loaded'
        events.timestamp[events.timestamp < '2016-05-01'] = '2016-05-01 00:00:00'
        events.timestamp[events.timestamp > '2016-05-08'] = '2016-05-07 23:59:59'
        events['hour'] = pd.to_datetime(events.timestamp).dt.hour
        events_agg_hourly = events.groupby(by = ['device_id','hour'], as_index = False)['event_id'].count()
        events_agg_hourly = events_agg_hourly.pivot_table(index = 'device_id', columns = 'hour', values = 'event_id', fill_value = 0)
        events_agg_hourly['no_rows'] = events.groupby(by = 'device_id', as_index = True)['event_id'].count()
        self.train = self.train.merge(events_agg_hourly, how = 'left', left_index = True, right_index = True).sort_values(by = ['trrow']).fillna(0)
        self.test = self.test.merge(events_agg_hourly, how = 'left', left_index = True, right_index = True).sort_values(by = ['terow']).fillna(0)
        self.xtr_evnt = csr_matrix(self.train.ix[:,-25:])
        self.xte_evnt = csr_matrix(self.test.ix[:,-25:])
        print 'Event level matrix created with Train : %d Rows and %d Cols; Test : %d Rows and %d Cols' %(self.xtr_evnt.shape[0],self.xtr_evnt.shape[1], self.xte_evnt.shape[0], self.xte_evnt.shape[1])
        self.train, self.test = self.train[['trrow',self.dv]],self.test[['terow',self.dv]]
        return events
        
    def get_app_labels(self, events):
        # Import app_events data
        app_evnt = pd.read_csv(os.path.join(wd,'Data', '.'.join(['app_events','csv'])))
        print 'App Events data loaded'
        app_evnt['app'] = self.encoder_app.fit_transform(app_evnt.app_id)
        napps = len(self.encoder_app.classes_) # Finding number of classes
        print 'Number of apps : %d' %(napps)
        # Joining the events and app events data
        dev_evnt = events[['event_id','device_id']].merge(app_evnt[['event_id','app']], how = 'inner', on = 'event_id').groupby(['device_id','app'])['app'].agg(['size']).merge(self.train[['trrow']], how = 'left' , left_index = True, right_index = True).merge(self.test[['terow']], how = 'left' , left_index = True, right_index = True).reset_index()
        d = dev_evnt.dropna(subset=['trrow'])
        self.Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trrow, d.app)), shape=(self.train.shape[0],napps))
        d = dev_evnt.dropna(subset=['terow'])
        self.Xte_app = csr_matrix((np.ones(d.shape[0]), (d.terow, d.app)), shape=(self.test.shape[0],napps))
        print 'App matrix created with Train : %d Rows and %d Cols; Test : %d Rows and %d Cols' %(self.Xtr_app.shape[0],self.Xtr_app.shape[1], self.Xte_app.shape[0], self.Xte_app.shape[1])

        # Get the app labels data
        applabels = pd.read_csv(os.path.join(wd,'Data', '.'.join(['app_labels','csv'])))
        print 'App Lables data loaded'
        applabels = applabels.loc[applabels.app_id.isin(app_evnt.app_id.unique())]
        applabels['app'] = self.encoder_app.transform(applabels.app_id)
        applabels['label'] = self.encoder_label.fit_transform(applabels.label_id)
        nlabels = len(self.encoder_label.classes_)
        print 'Number of labels : %d' %(nlabels)

        # Obtaining bag of labels
        devicelabels = (dev_evnt[['device_id','app']].merge(applabels[['app','label']]).groupby(['device_id','label'])['app'].agg(['size']).merge(self.train[['trrow']], how='left', left_index=True, right_index=True).merge(self.test[['terow']], how='left', left_index=True, right_index=True).reset_index())
        d = devicelabels.dropna(subset=['trrow'])
        self.Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trrow, d.label)), shape=(self.train.shape[0],nlabels))
        d = devicelabels.dropna(subset=['terow'])
        self.Xte_label = csr_matrix((np.ones(d.shape[0]), (d.terow, d.label)), shape=(self.test.shape[0],nlabels))
        print 'Label matrix created with Train : %d Rows and %d Cols; Test : %d Rows and %d Cols' %(self.Xtr_label.shape[0],self.Xtr_label.shape[1], self.Xte_label.shape[0], self.Xte_label.shape[1])

    def get_dv(self):
        self.y_tr = self.encoder_dv.fit_transform(self.train.sort_values(by =['trrow'])[[self.dv]])
        if self.action == 'GridSearch':
            self.y_te = self.encoder_dv.transform(self.test.sort_values(by = ['terow'])[[self.dv]])
            self.y_te = np.array([x[0] for x in self.y_te])
        
        
class MakePredictions(TrainTest):
    '''
    This function can fit the data, cross validate on validation sample and write the submission to a file with time stamp attached
    Parameters to pass:
    Mandatory
    action --> 'GridSearch', 'Prediction'
    
    Optional
    dv --> default 'group'
    train, test --> Default will call load_data if not provided
    '''
    def __init__ (self,action, dv = 'group', fileName = None):
        super(MakePredictions, self).__init__(action = action, dv = dv)
        super(MakePredictions, self).load_data()
        #self.features = list(self.train.columns)
        self.fileName = fileName
        self.preds_test = None
        
    def fit(self, num_boost_round = 200, early_stopping_rounds = 50, getImportance = False, **params):
        self.params = params['params']
        if self.action == 'Prediction':
            train, earlyStop, y_tr, earlyStop_y = train_test_split(self.train, self.y_tr, test_size = 3000, random_state = 42)
            dtrain = xgb.DMatrix(train,y_tr)
            dEarlyStop = xgb.DMatrix(earlyStop,earlyStop_y)
            watchlist = [(dtrain, 'train'), (dEarlyStop, 'eval')]
        if self.action == 'GridSearch':
            dtrain = xgb.DMatrix(self.train, self.y_tr)
            dtest = xgb.DMatrix(self.test,self.y_te)
            watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        self.model = xgb.train(self.params, dtrain, num_boost_round,verbose_eval=False, evals=watchlist, early_stopping_rounds=early_stopping_rounds)
        self.preds_test = None # To Predict again when score and writefile is called
        if getImportance:
            self.ceate_feature_map() # Creating feature map file for feature importance
            self.importance = self.model.get_fscore(fmap = 'xgb.fmap')
        
    def XgbPredict(self):
        dtest = xgb.DMatrix(self.test)
        self.preds_test = self.model.predict(dtest, ntree_limit=self.model.best_iteration)
        if self.action == 'GridSearch':
            dtrain = xgb.DMatrix(self.train, self.y_tr)
            self.preds_train = self.model.predict(dtrain, ntree_limit=self.model.best_iteration)

    def score(self):
        if self.preds_test is None:
            self.XgbPredict()
        self.score_test = log_loss(self.y_te.tolist(), self.preds_test)
        self.score_train = log_loss(self.y_tr.tolist(), self.preds_train)

    def WriteFile(self):
        if self.preds_test is None:
            self.XgbPredict()
        toWrite = pd.DataFrame(self.preds_test, columns = list(self.encoder_dv.classes_))
        toWrite['device_id'] = self.fileIndex
        cols = toWrite.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        toWrite = toWrite[cols]
        if not self.fileName:
             self.fileName = 'submission_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        toWrite.to_csv(os.path.join(wd, '.'.join([self.fileName,'csv'])),index = False) 
        
    def ceate_feature_map(self):
        '''
        This stores the feature map .fmap file to be used to obtain feature importance in XGBoost
        '''
        outfile = open('xgb.fmap', 'w')
        i = 0
        for feat in self.features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
        outfile.close()

# Other functions
def oneHotEncode(df, feats):
    '''
    Takes a data set and columns needed to encode
    Removes the original columns, encodes those one and appends and returns the data set
    '''
    dummies = pd.get_dummies(df[feats], dummy_na = True, columns = feats) 
    df = df.drop(feats, axis = 1, inplace = False)
    return (df.join(dummies))


