import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from catboost import Pool
from .data_utils import gen_features_share, gen_features_price, anomaly_detect

DROP_COLS_SHARE = ['key_rate_diff',  'objectid', 'S_volume', 
                   'S_share_shift4_diff', 'S_share_shift5_diff', 'S_share_shift3_diff3',
                   'key_rate-key_rate_mean_6']
DROP_COLS_PRICE = ['objectid', 'key_rate-key_rate_mean_6', 'key_rate-key_rate_mean_12', 'months_after_open',
                   'S_share_shift3_diff2', 'S_share_shift3_diff3', 'combine_price_shift3_diff6', 'combine_price_shift3_diff12',
                   'mean_price_shift3_diff6', 'mean_price_shift3_diff12',
                         ]



def make_model_forecast(df, kr, first_forecast_date, y_col, n_quartals, model,
                        date_col = 'report_date', fit = True, return_last_train = True):

    drop_cols = [date_col]
    drop_cols.extend(DROP_COLS_SHARE if y_col == 'S_share' else DROP_COLS_PRICE)
    index_cols = ['objectid', date_col]
    target_cols = [y_col]
    
    gen_features = gen_features_share if y_col == 'S_share' else gen_features_price
    
    if fit:
        train_df = gen_features(df, kr)
        train_df = train_df[train_df[date_col] < first_forecast_date]
        X_train, y_train = train_df.drop(columns = [*drop_cols, *target_cols]), train_df[target_cols]
        train_pool = Pool(data=X_train, label=y_train)
        model.fit(train_pool, verbose = False)
    
    left_forecast_date = first_forecast_date
    if y_col == 'target_price_pm':
        target = gen_features(df, kr).loc[df['report_date'] <= first_forecast_date, [*index_cols, y_col]]
        df = df.merge(target, on = index_cols, how = 'left')
    else:
        df = df.copy()

    for q in range(n_quartals):
        df_q = gen_features(df, kr, (q+1)*3)
        test_df_q = df_q[df_q[date_col].between(left_forecast_date, left_forecast_date + MonthEnd(2))]
        X_test_q, X_test_index_cols = test_df_q.drop(columns = [*drop_cols, *target_cols]), test_df_q[index_cols]
        test_pool_q = Pool(data=X_test_q)
        y_pred = model.predict(test_pool_q)
        y_pred = X_test_index_cols.assign(**{y_col: y_pred})
        
        if y_col == 'S_share':
            y_pred = y_pred.groupby('objectid').apply(lambda df: df.assign(**{y_col: 
                            np.minimum(1, df.sort_values('report_date')[y_col].cummax())})).reset_index(drop = True)
        
        df = df.merge(y_pred, on = ['objectid', date_col], how = 'left')
        df.loc[df[date_col].isin(X_test_index_cols[date_col]), 'forecast_step'] = q + 1
        
        df[y_col] = df[f'{y_col}_x'].fillna(df[f'{y_col}_y'])
        df = df.drop(columns = [f'{y_col}_x', f'{y_col}_y'])
            
        left_forecast_date += MonthEnd(3)
    
    cols_to_return = ['forecast_step', *index_cols, y_col, 'S', 'close_date', 'escrow_volume']
    cols_to_return.extend(['S_share'] if y_col == 'target_price_pm' else [])
    df = df.loc[(df[date_col] >= (first_forecast_date - MonthEnd(1) if return_last_train else first_forecast_date))&
                (df[date_col] <= first_forecast_date + MonthEnd(3*n_quartals)), cols_to_return].fillna({'forecast_step':0})
    
    if y_col == 'S_share':
        df = df.groupby('objectid').apply(lambda df: df.assign(**{y_col: 
                            np.minimum(1, df.sort_values('report_date')[y_col].cummax())})).reset_index(drop = True)
        #TODO: в конце срока эксплуатации сделать долю равной 1 (опционально)
      
    return df


def make_escrow_volume_forecast(df_share, df_price, kr, first_forecast_date, n_quartals, model_share, model_price, 
                                                date_col = 'report_date'):
    y_share = 'S_share'
    y_price = 'target_price_pm'
    fill_cols = ['S_share', 'S', 'close_date', 'escrow_volume']
    
    df_share = df_share.sort_values(date_col).groupby('objectid').apply(lambda df: df.assign(
                            escrow_volume = df['escrow_volume'].ffill())
                                      ).reset_index(drop = True)
    
    share_pred = make_model_forecast(df_share, kr, first_forecast_date, y_share, n_quartals, model_share)
    df_price = df_price.merge(share_pred, on = [date_col, 'objectid'], how = 'outer')
    for col in fill_cols:
        df_price.loc[:, col] = df_price.loc[:, f'{col}_x'].fillna(df_price.loc[:, f'{col}_y']) 
    df_price = df_price.drop(columns = [f'{col}_{post}' for col in fill_cols for post in ['x', 'y']])
    fcst = make_model_forecast(df_price, kr, first_forecast_date, y_price, n_quartals, model_share)
    fcst.loc[:, 'S_volume'] = fcst.loc[:, 'S_share']*fcst.loc[:, 'S']
    fcst = fcst.sort_values(date_col).groupby('objectid').apply(lambda df: df.assign(
            escrow_volume = df['S_volume'].diff().fillna(0)*df['target_price_pm']+df['escrow_volume'].values[0])
                                      ).reset_index(drop = True)
    
    return fcst



