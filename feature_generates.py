import pandas as pd
import numpy as np
from tqdm.notebook import tqdm


# Фичи из файла chronom
def chronom_read(data_path='data_task1', types='TRAIN'):
    chronom = pd.read_csv(
        f'{data_path}/chronom_{types.lower()}.csv').drop(['Unnamed: 0'], axis=1)
    chronom['VR_KON'] = pd.to_datetime(chronom['VR_KON'])
    chronom['VR_NACH'] = pd.to_datetime(chronom['VR_NACH'])
    chronom['time'] = (chronom['VR_KON'] - chronom['VR_NACH']).dt.seconds
    chronom['hour'] = (chronom['VR_NACH']).dt.hour
    chronom['minute'] = (chronom['VR_NACH']).dt.minute
    chronom_new = pd.pivot_table(chronom,
                                 index='NPLV', columns=['TYPE_OPER', "NOP"],
                                 aggfunc='mean')
    new_cols_name = []
    for j in range(chronom_new.shape[1]):
        new_cols_name.append(" ".join(str(i) for i in chronom_new.columns[j]))
    chronom_new.columns = new_cols_name
    chronom_new.reset_index(inplace=True)
    chronom_new = chronom_new.merge(
        chronom[chronom.VR_NACH > '2020-01-01'].groupby(['NPLV']).agg({'VR_KON': 'max', 'VR_NACH': 'min'}), on='NPLV')
    chronom_new.columns = [
        'chronom_' +
        str(j) if i > 0 and i < chronom_new.shape[1] -
        2 else j for i,
        j in enumerate(
            chronom_new.columns)]
    return chronom_new


# Фичи из файла chugun
def chugun_read(data_path='data_task1', types='TRAIN'):
    chugun = pd.read_csv(f'{data_path}/chugun_{types.lower()}.csv')
    chugun['DATA_ZAMERA'] = pd.to_datetime(chugun['DATA_ZAMERA'])
    chugun.columns = [
        'chugun_' +
        str(j) if i > 0 and i < chugun.shape[1] -
        1 else j for i,
        j in enumerate(
            chugun.columns)]
    return chugun


# Фичи из файла produv
def produv_read(data_path='data_task1', types='TRAIN'):
    produv = pd.read_csv(f'{data_path}/produv_{types.lower()}.csv')
    for i in ['RAS', 'POL']:
        for d in [2, 4, 6, 10, 25, 50, 100, 500, 1000]:
            produv[f'{i}_diff_{d}'] = produv.groupby(['NPLV'])[i].diff(d)
    produv_new = produv.merge(produv.groupby(['NPLV'])['RAS'].rolling(3).max().to_frame('RAS_3_max').reset_index(),
                              left_index=True, right_on='level_1').drop(['level_1', 'NPLV_y'], axis=1).rename(columns={'NPLV_x': 'NPLV'})\
        .merge(
        produv.groupby(['NPLV'])['POL'].rolling(
            3).max().to_frame('POL_3_max').reset_index(),
        left_index=True, right_on='level_1').drop(['level_1', 'NPLV_y'], axis=1).rename(columns={'NPLV_x': 'NPLV'})\
        .merge(
        produv.groupby(['NPLV'])['RAS'].rolling(
            3).mean().to_frame('RAS_3_mean').reset_index(),
        left_index=True, right_on='level_1').drop(['level_1', 'NPLV_y'], axis=1).rename(columns={'NPLV_x': 'NPLV'})\
        .merge(
            produv.groupby(['NPLV'])['POL'].rolling(
                3).mean().to_frame('POL_3_mean').reset_index(),
            left_index=True, right_on='level_1').drop(['level_1', 'NPLV_y'], axis=1).rename(columns={'NPLV_x': 'NPLV'})\
        .merge(
                produv.groupby(['NPLV'])['RAS'].rolling(
                    10).max().to_frame('RAS_10_max').reset_index(),
            left_index=True, right_on='level_1').drop(['level_1', 'NPLV_y'], axis=1).rename(columns={'NPLV_x': 'NPLV'})\
        .merge(
        produv.groupby(['NPLV'])['POL'].rolling(
            10).max().to_frame('POL_10_max').reset_index(),
        left_index=True, right_on='level_1').drop(['level_1', 'NPLV_y'], axis=1).rename(columns={'NPLV_x': 'NPLV'})\
        .merge(
        produv.groupby(['NPLV'])['RAS'].rolling(
            10).mean().to_frame('RAS_10_mean').reset_index(),
        left_index=True, right_on='level_1').drop(['level_1', 'NPLV_y'], axis=1).rename(columns={'NPLV_x': 'NPLV'})\
        .merge(
        produv.groupby(['NPLV'])['POL'].rolling(
            10).mean().to_frame('POL_10_mean').reset_index(),
        left_index=True, right_on='level_1').drop(['level_1', 'NPLV_y'], axis=1).rename(columns={'NPLV_x': 'NPLV'})\
        .merge(
                produv.groupby(['NPLV'])['RAS'].rolling(
                    25).max().to_frame('RAS_25_max').reset_index(),
            left_index=True, right_on='level_1').drop(['level_1', 'NPLV_y'], axis=1).rename(columns={'NPLV_x': 'NPLV'})\
        .merge(
        produv.groupby(['NPLV'])['POL'].rolling(
            25).max().to_frame('POL_25_max').reset_index(),
        left_index=True, right_on='level_1').drop(['level_1', 'NPLV_y'], axis=1).rename(columns={'NPLV_x': 'NPLV'})\
        .merge(
        produv.groupby(['NPLV'])['RAS'].rolling(
            25).mean().to_frame('RAS_25_mean').reset_index(),
        left_index=True, right_on='level_1').drop(['level_1', 'NPLV_y'], axis=1).rename(columns={'NPLV_x': 'NPLV'})\
        .merge(
        produv.groupby(['NPLV'])['POL'].rolling(
            25).mean().to_frame('POL_25_mean').reset_index(),
        left_index=True, right_on='level_1').drop(['level_1', 'NPLV_y'], axis=1).rename(columns={'NPLV_x': 'NPLV'})
    produv_new2 = produv_new.groupby(['NPLV']).agg(
        ['mean', 'max', 'median', 'last', 'std'])

    new_cols_name = []
    for j in range(produv_new2.shape[1]):
        new_cols_name.append(" ".join(str(i) for i in produv_new2.columns[j]))
    produv_new2.columns = new_cols_name
    produv_new2.columns = [
        'produv_' + str(j) for i,
        j in enumerate(
            produv_new2.columns)]
    return produv_new2.reset_index()


# Фичи из файла gas
def gas_read(data_path='data_task1', types='TRAIN'):
    gas1 = pd.read_csv(f'{data_path}/gas_{types.lower()}.csv')
    for i in gas1.columns[2:]:
        for d in [2, 4, 6, 10, 25, 50, 100, 500, 1000]:
            gas1[f'{i}_diff_{d}'] = gas1.groupby(['NPLV'])[i].diff(d)
    nplvs = gas1.NPLV.unique()
    gas_new2 = pd.DataFrame()
    for nplv in tqdm(nplvs):
        gas = gas1[gas1.NPLV == nplv].copy()
        for i in (gas.columns[3:13]):
            gas_new = gas.merge(gas.groupby(['NPLV'])[i].rolling(3).max().to_frame(f'{i}_3_max').reset_index(),
                                left_index=True, right_on='level_1', how='left').drop(['level_1', 'NPLV_y'], axis=1)\
                .rename(columns={'NPLV_x': 'NPLV'}).merge(
                gas.groupby(['NPLV'])[i].rolling(
                    3).mean().to_frame(f'{i}_3_mean').reset_index(),
                left_index=True, right_on='level_1', how='left').drop(['level_1', 'NPLV_y'], axis=1).rename(columns={'NPLV_x': 'NPLV'})\
                .merge(gas.groupby(['NPLV'])[i].rolling(10).max().to_frame(f'{i}_10_max').reset_index(),
                       left_index=True, right_on='level_1', how='left').drop(['level_1', 'NPLV_y'], axis=1)\
                .rename(columns={'NPLV_x': 'NPLV'}).merge(
                gas.groupby(['NPLV'])[i].rolling(10).mean().to_frame(
                    f'{i}_10_mean').reset_index(),
                left_index=True, right_on='level_1', how='left').drop(['level_1', 'NPLV_y'], axis=1)\
                .rename(columns={'NPLV_x': 'NPLV'})
        gas_new_res = gas_new.groupby(['NPLV']).agg(
            ['mean', 'max', 'median', 'last', 'std'])
        gas_new2 = pd.concat([gas_new2, gas_new_res])
    new_cols_name = []
    for j in range(gas_new2.shape[1]):
        new_cols_name.append(" ".join(str(i) for i in gas_new2.columns[j]))
    gas_new2.columns = new_cols_name
    gas_new2.columns = [
        'gas_' + str(j) for i,
        j in enumerate(
            gas_new2.columns)]
    return gas_new2.reset_index()


# Фичи из файла lom
def lom_read(data_path='data_task1', types='TRAIN'):
    lom = pd.read_csv(f'{data_path}/lom_{types.lower()}.csv')
    lom = pd.pivot_table(
        lom,
        index='NPLV',
        columns='VDL',
        values='VES').reset_index()
    lom.columns = [
        'lom_' +
        str(j) if j != 'NPLV' else j for i,
        j in enumerate(
            lom.columns)]
    return lom


# Фичи из файла plavki
def plavki_read(data_path='data_task1', types='TRAIN'):
    plavki = pd.read_csv(f'{data_path}/plavki_{types.lower()}.csv')
    plavki['plavki_time'] = (
        pd.to_datetime(
            plavki['plavka_VR_KON']) -
        pd.to_datetime(
            plavki['plavka_VR_NACH'])).dt.seconds
    plavki.columns = ['lom_' + str(j) if j not in ['NPLV', 'plavka_VR_NACH', 'plavka_VR_KON']
                      else j for i, j in enumerate(plavki.columns)]
    return plavki


# Фичи из файла sip
def sip_read(data_path='data_task1', types='TRAIN'):
    sip = pd.read_csv(f'{data_path}/sip_{types.lower()}.csv')
    sip['DAT_OTD'] = pd.to_datetime(sip['DAT_OTD'])
    sip = sip.merge(
        sip.groupby(
            ['NPLV'])['DAT_OTD'].min().to_frame('min_time'),
        left_on='NPLV',
        right_index=True)
    sip['time'] = (sip['DAT_OTD'] - sip['min_time']).dt.seconds
    sip2 = pd.pivot_table(
        sip,
        index='NPLV',
        columns=['NMSYP'],
        values=[
            'VSSYP',
            'time'],
        aggfunc='median')
    new_cols_name = []
    for j in range(sip2.shape[1]):
        new_cols_name.append("_".join(str(i) for i in sip2.columns[j]))
    sip2.columns = new_cols_name
    sip2.columns = ['sip_' + str(j) for i, j in enumerate(sip2.columns)]
    return sip2


# Несколько фичей по времени
def time_plavki_feats(all_df):
    all_df['DATA_ZAMERA_CHUGUN_start'] = (
        all_df['DATA_ZAMERA'] -
        all_df['VR_NACH']).dt.seconds
    all_df['DATA_ZAMERA_CHUGUN_end'] = (
        all_df['VR_KON'] - all_df['DATA_ZAMERA']).dt.seconds
    all_df['DATA_ZAMERA'] = all_df['DATA_ZAMERA_CHUGUN_start'] / (all_df['DATA_ZAMERA_CHUGUN_start'] +
                                                                  all_df['DATA_ZAMERA_CHUGUN_end'])
    all_df['plavki_end_from_start'] = (
        pd.to_datetime(
            all_df['plavka_VR_KON']) -
        all_df['VR_NACH']).dt.seconds
    all_df['plavki_end_from_end'] = (
        pd.to_datetime(
            all_df['plavka_VR_KON']) -
        all_df['VR_KON']).dt.seconds
    all_df['plavki_start_from_start'] = (
        pd.to_datetime(
            all_df['plavka_VR_NACH']) -
        all_df['VR_NACH']).dt.seconds
    all_df['plavki_start_from_end'] = (
        pd.to_datetime(
            all_df['plavka_VR_NACH']) -
        all_df['VR_KON']).dt.seconds
    all_df['plavki_start_share'] = all_df['plavki_start_from_start'] / (all_df['plavki_start_from_start']
                                                                        + all_df['plavki_start_from_end'])
    all_df['plavki_end_share'] = all_df['plavki_end_from_start'] / (all_df['plavki_end_from_start']
                                                                    + all_df['plavki_end_from_end'])
    return all_df


# Агрегируем все фичи
def read_all(data_path='data_task1', types='TRAIN'):
    print("STEP 1 of 7: Chronom features added")
    chr = chronom_read(data_path, types)
    print("STEP 2 of 7: Chugun features added")
    chugun = chugun_read(data_path, types)
    df = chr.merge(chugun, on='NPLV')
    print("STEP 3 of 7: Produv features added (~4 minutes for train)")
    produv = produv_read(data_path, types)
    all_df = produv.merge(df, on=['NPLV'], how='left')
    all_df = time_plavki_feats(all_df)
    print(all_df.shape)
    print("STEP 4 of 7: LOM features added")
    lom = lom_read(data_path, types)
    print("STEP 5 of 7: Plavki features added")
    plavki = plavki_read(data_path, types)
    print("STEP 6 of 7: SIP features added")
    sip = sip_read(data_path, types)
    print("STEP 7 of 7: GAS features added (~20 minutes for train)")
    gas_feat = gas_read(data_path, types)
    all_df = all_df.merge(gas_feat, on='NPLV', how='left')\
        .merge(lom, on='NPLV', how='left').merge(plavki, on='NPLV', how='left')\
        .merge(sip, on='NPLV', how='left')

    if types == 'TRAIN':
        target = pd.read_csv(f'{data_path}/target_train.csv')
        all_df = all_df.merge(target, on=['NPLV'])
    all_df = time_plavki_feats(all_df)
    print(all_df.shape)
    return all_df
