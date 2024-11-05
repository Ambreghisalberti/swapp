from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import os
from scipy.signal import medfilt
__HERE__ = os.path.dirname(__file__)

def standardScaling(df):
    """ Precondition : df has to be very big, so the scaling is close to the one of the traintest of the model.
    The best would have been to use the scaler used during training."""
    scaler = StandardScaler()
    scaled_all_data = df.copy()
    scaled_all_data.loc[:, :] = scaler.fit_transform(df.values)
    return scaled_all_data


def pred(df, model):
    scaled_df = standardScaling(df)
    '''
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(scaled_df)
    filled_values = imp.transform(scaled_df)
    predictions = model.predict(filled_values)  # Can take about 3 minutes for the whole dataset
    df.loc[:, 'regions_pred'] = medfilt(predictions, kernel_size=5).astype(int)
    '''
    predictions = model.predict(scaled_df.dropna().values)  # Can take about 3 minutes for the whole dataset
    df['regions_pred'] = np.nan
    df.loc[df.dropna().index.values, 'regions_pred'] = medfilt(predictions, kernel_size=5).astype(int)


def pred_boosting(df, **kwargs):
    # Works only in 1.2.2 version of scikit learn for MMS, 1.0.2 for themis
    # run f"pip install scikit-learn=={version}" before if needed
    if 'model' in kwargs:
        model = kwargs['model']
    else:
        model = pd.read_pickle(f"{__HERE__}/boosting.pkl")
    pred(df, model)
