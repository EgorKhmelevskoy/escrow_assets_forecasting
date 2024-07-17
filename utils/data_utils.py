import numpy as np
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import MonthEnd

def gen_features_share(df, kr, ffill_kr_months = 0, return_selected_features = True):
    """
        Функция генерации фичей для модели прогноза доли проданных квадратных метров.
        
        Parameters
        ----------
        df: pd.DataFrame
            Датафрейм с данными по эскроу
        kr: pd.DataFrame
            Датафрейм с данными по ключевой ставке ЦБ РФ
        ffill_kr_months: int, default = 0 
            Насколько месяцев вперед продлить последнее значение ключевой ставки 
        return_selected_features: bool, default = True   
            Параметр того, нужно ли возвращать датафрейм с выбранными колонками
        Returns
        -------
        df : pd.DataFrame
            Датафрейм c фичами и таргетом для модели прогноза доли проданных квадратных метров
    """
    
    kr = kr.assign(report_date = kr['Дата'].map(lambda dt: datetime(*[int(dt_) for dt_ in reversed(dt.split('.'))], 1) + MonthEnd(0))).rename(
                                                            columns = {'Ключевая ставка, % годовых': 'key_rate'}).sort_values('report_date')
    if ffill_kr_months:

        kr = pd.concat([kr, pd.DataFrame({'report_date': pd.date_range(kr['report_date'].max() + MonthEnd(1), 
                                         kr['report_date'].max() + MonthEnd(ffill_kr_months), freq = 'M'),
                                          'key_rate': kr.loc[kr['report_date'] == kr['report_date'].max(), 'key_rate'].values[0]})])                                
    kr = kr.sort_values('report_date').assign(key_rate_diff = kr['key_rate'].diff().fillna(0))
    kr = kr.assign(**{f'key_rate-key_rate_mean_{period}': kr['key_rate'] - kr['key_rate'].shift(1).rolling(period).mean() for period in [3, 6, 12]})
    kr = kr[['report_date', *[col for col in kr.columns if col.startswith('key')]]]
    df = df.merge(kr, on = 'report_date', how = 'left')
    get_normalized = lambda arr: (arr - np.mean(arr)) / np.std(arr, ddof = 1)
    df = df.sort_values('report_date').groupby('objectid').apply(lambda df: df.assign(months_num = df.shape[0],
                                                                                      months_after_open = get_normalized(np.arange(0, df.shape[0])),
                                                                                      months_before_close = get_normalized(np.arange(df.shape[0]-1, -1, -1)),
                                                                                      share_months_to_close = np.arange(0, df.shape[0])/(df.shape[0] - 1),
                                                                                      december_nums = df['report_date'].map(lambda dt: dt.month == 12).cumsum(),
                                            S_share_shift3 = df['S_share'].shift(3).fillna(0),#.fillna(df['S_share'].shift(2)).fillna(df['S_share'].shift(1)).fillna(df['S_share'])
                                            S_share_shift3_diff_mean3 = df['S_share'].shift(3).diff().rolling(3).mean().fillna(0),
                                            S_share_shift3_diff = df['S_share'].shift(3).diff().fillna(0),
                                            S_share_shift4_diff = df['S_share'].shift(4).diff().fillna(0),
                                            S_share_shift5_diff = df['S_share'].shift(5).diff().fillna(0),
                                            S_share_shift3_diff2 = df['S_share'].shift(3).diff(2).fillna(0),   
                                            S_share_shift3_diff3 = df['S_share'].shift(3).diff(3).fillna(0), 
                                                                                     )).reset_index(drop = True)
    feature_cols = ['report_date', 'objectid', 'S', 'december_nums', 
                    *[col for col in df.columns if col.startswith('key') or ('months' in col) or ('shift' in col)]]
    target_cols = ['S_share', 'S_volume']
    
    if return_selected_features:
        df = df[[*feature_cols, *target_cols]]
        
    return df




def gen_features_price(df, kr, ffill_kr_months = 0, return_selected_features = True, ln = False):
    """
        Функция генерации фичей для модели прогноза средней цены за квадратный метр для объекта недвижимости.
        
        Parameters
        ----------
        df: pd.DataFrame
            Датафрейм с данными по эскроу
        kr: pd.DataFrame
            Датафрейм с данными по ключевой ставке ЦБ РФ
        ffill_kr_months: int, default = 0 
            Насколько месяцев вперед продлить последнее значение ключевой ставки 
        return_selected_features: bool, default = True   
            Параметр того, нужно ли возвращать датафрейм с выбранными колонками
        ln: bool, default = True   
            Параметр того, нужно ли логарифмировать таргет и производные от него фичи
        Returns
        -------
        df : pd.DataFrame
            Датафрейм c фичами и таргетом для модели прогноза доли проданных квадратных метров
    """
    kr = kr.assign(report_date = kr['Дата'].map(lambda dt: datetime(*[int(dt_) for dt_ in reversed(dt.split('.'))], 1) + MonthEnd(0))).rename(
                                                            columns = {'Ключевая ставка, % годовых': 'key_rate'}).sort_values('report_date')
    if ffill_kr_months:

        kr = pd.concat([kr, pd.DataFrame({'report_date': pd.date_range(kr['report_date'].max() + MonthEnd(1), 
                                         kr['report_date'].max() + MonthEnd(ffill_kr_months), freq = 'M'),
                                          'key_rate': kr.loc[kr['report_date'] == kr['report_date'].max(), 'key_rate'].values[0]})])                                
    kr = kr.sort_values('report_date').assign(key_rate_diff = kr['key_rate'].diff().fillna(0))
    kr = kr.assign(**{f'key_rate-key_rate_mean_{period}': kr['key_rate'] - kr['key_rate'].shift(1).rolling(period).mean() for period in [3, 6, 12]})
    kr = kr[['report_date', *[col for col in kr.columns if col.startswith('key')]]]
    
    df = df.groupby('objectid').apply(lambda df: anomaly_detect(df, ['mean_price'], True))
    df = df.merge(kr, on = 'report_date', how = 'left')
    price_cols = ['mean_price', 'mean_price_per_metr']
    df = df.sort_values('report_date').groupby('objectid').apply(lambda df: df.assign(**{rd_col: df[col].interpolate().ffill().bfill() for 
                                                                                         col, rd_col in zip(price_cols, ['mean_price', 'target_price_pm'])},
                                                    combine_price = df['mean_price_per_metr'].interpolate().fillna(df['mean_price']).ffill().bfill()
                                                                                     )
                                                                ).reset_index(drop = True)
    get_normalized = lambda arr: (arr - np.mean(arr)) / np.std(arr, ddof = 1)
    df = df.sort_values('report_date').groupby('objectid').apply(lambda df: df.assign(months_num = df.shape[0],
                                                                                      months_after_open = get_normalized(np.arange(0, df.shape[0])),
                                                                                      months_before_close = get_normalized(np.arange(df.shape[0]-1, -1, -1)),
                                                                                      share_months_to_close = np.arange(0, df.shape[0])/(df.shape[0] - 1),
                                                                                      december_flg = (df['report_date'].map(lambda dt: dt.month == 12)).astype('int'),
                                                                                      december_nums = df['report_date'].map(lambda dt: dt.month == 12).cumsum(),
                                                                                      **{f'{col}_shift3': df[col].shift(3).bfill() for col in ['combine_price', 'mean_price']},
                                                                    
                                            S_share_shift_3 = df['S_share'].shift(3).fillna(0),#.fillna(df['S_share'].shift(2)).fillna(df['S_share'].shift(1)).fillna(df['S_share'])
                                            S_share_shift3_diff_mean3 = df['S_share'].shift(3).diff().rolling(3).mean().fillna(0),
                                            S_share_shift3_diff = df['S_share'].shift(3).diff().fillna(0),
                                            S_share_shift3_diff2 = df['S_share'].shift(3).diff(2).fillna(0),   
                                            S_share_shift3_diff3 = df['S_share'].shift(3).diff(3).fillna(0),                                        
                                                                                     )).reset_index(drop = True)
    
    f = lambda df: np.log1p(df) if ln else df
        
    df = df.sort_values('report_date').groupby('objectid').apply(lambda df: df.assign(**{f'{col}_shift3_diff{period}': (f(df[col].shift(3)) - f(df[col].shift(4).rolling(period).mean())).fillna(0)
                                                                                          for col in ['combine_price', 'mean_price'] for period in [1, 3, 6, 12]})).reset_index(drop = True)
    
    df = df.fillna({'mean_price_shift3': df['combine_price_shift3']}) # для некоторых объектов не нашлось ближайших :(                                                                            
    feature_cols = ['report_date', 'objectid', 'december_nums',
                    *[col for col in df.columns if col.startswith('key') or ('shift' in col) or ('months' in col)]]
    target_cols = ['target_price_pm']
    
    if ln:
        df = df.assign(**{col: np.log1p(df[col]) for col in df.columns if ('price' in col) and ('diff' not in col)})
        
    if return_selected_features:
        df = df[[*feature_cols, *target_cols]]
        
    return df


def anomaly_detect(df, cols, fill = False):
    for col in cols:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        IQR = q75 - q25
        if fill:
            get_val = lambda arr, f: arr.agg(f) if len(arr) > 0 else None
            df.loc[:, col] = df[col].mask(df[col] < q25 - 1.5*IQR, get_val(df[col][df[col] >= q25 - 1.5*IQR], 'min'))
            df.loc[:, col] = df[col].mask(df[col] > q75 + 1.5*IQR, get_val(df[col][df[col] <= q75 + 1.5*IQR], 'max'))
        else:   
            df = df[df[col].between(q25 - 1.5*IQR, q75 + 1.5*IQR)]
    return df

