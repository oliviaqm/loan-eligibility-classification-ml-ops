from flask import Flask, request
from fancyimpute import KNN, SoftImpute

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.preprocessing import LabelBinarizer,StandardScaler,OrdinalEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.stats import boxcox
from sklearn.linear_model import LogisticRegression,RidgeClassifier, PassiveAggressiveClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib
import json
from werkzeug.utils import secure_filename

import traceback
import operator
import six
import sys
sys.modules['sklearn.externals.six'] = six
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn.utils import _safe_indexing
sys.modules['sklearn.utils.safe_indexing'] = sklearn.utils._safe_indexing

from imblearn.over_sampling import SMOTE

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello Mate'

def check_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ["csv"]

@app.route('/eligibility', methods=['POST'])
def eligibility_check():
    if request.method == "POST":
        try:
            
            if 'csv' not in request.files:
                return {
                    "code": 404,
                    "msg": "Csv Not Found."
                }
            file = request.files["csv"]
            if file.filename == "":
                return {
                    "code": 404,
                    "msg": "Csv Not Found."
                }

            if file and check_file(file.filename):
                filename = secure_filename(file.filename)        

                test=pd.read_csv(file)
                cat_cols = ['Term','Years in current job','Home Ownership','Purpose']

                for c in cat_cols:
                    test[c] = pd.factorize(test[c])[0]

                #Imputing missing data with soft impute
                updated_test_data=pd.DataFrame(data=SoftImpute().fit_transform(test[test.columns[3:19]],), columns=test[test.columns[3:19]].columns, index=test.index)

                #Getting the dataset ready pd.get dummies function for dropping the dummy variables
                test_data = pd.get_dummies(updated_test_data, drop_first=True)

                gbm_pickle = joblib.load('GBM_Model_version1.pkl')

                y_pred = gbm_pickle.predict(test_data)

                y_pred = gbm_pickle.predict_proba(test_data)

                y_pred_1=np.where(y_pred ==0, 'Loan Approved', 'Loan Rejected')

                test['Loan Status']=y_pred_1

                # Get the output json

                out_data = test.replace({np.nan: None})
                json_data = out_data.to_dict('records')

                test.to_csv('Output_Test.csv',index=False)

                test = test.to_dict('records')



                
                return {
                    "code": "200",
                    "msg": "Fetched Successfully",
                    "resutls": json.loads(json.dumps(json_data))
                }
            
            else:
                return {
                    "code": 500,
                    "msg": "Something went wrong"
                }
        except:
            
            return {
                    "code": 500,
                    "msg": traceback.format_exc(),
                }



if __name__ == "__main__":
     app.run( debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)) )
         label="Regression_model", linewidth=2) 
