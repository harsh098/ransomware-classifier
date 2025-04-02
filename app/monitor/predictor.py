import sqlite3
import os
import time
import pandas as pd
from joblib import load
import pandas as pd
import numpy as np
import re

DB_PATH = os.getenv("LOGS_DB_NAME", "../../db.sqlite3")
FEATURES_PATH = os.getenv("FEATURES_PATH" , '../../data/features.joblib')
SCALER_PATH = os.getenv("SCALER_PATH" , '../../data/scaler.joblib')
MODEL_PATH = os.getenv("MODEL_PATH", '../../data/model.joblib_ensemble')
TYPE_NAMES = ['O', 'C', 'D', 'E']
TIME_PERIOD = 1e9

feature_names = load(FEATURES_PATH)
scaler = load(SCALER_PATH)
model = load(MODEL_PATH)

def counts(df):
    # group by PID, TYPE and PERIOD
    ts = df['TS']
    df1 = df.assign(PERIOD=np.trunc((ts - ts[0]) / TIME_PERIOD))
    df1.drop(columns=['TS', 'FLAG', 'OPEN', 'CREATE', 'DELETE', 'ENCRYPT', 'FILENAME'], inplace=True)

    # count the number of event grouped by type, period and PID and move TYPE to column
    grouped = df1.groupby(['TYPE', 'PERIOD', 'PID']).agg(['count','sum']).unstack(level='TYPE', fill_value=0)

    # aggregate over time period (max per period + total)
    aggregated = grouped.groupby(level='PID').agg(['max','sum'])

    # rename levels/columns (skip 'PATTERN')
    aggregated.columns = aggregated.columns.to_flat_index()
    aggregated.rename(columns={col: '_'.join(col[1:]) for col in aggregated.columns}, inplace=True)

    # sum the number of pattern matches across events
    pattern_max = re.compile("^sum_\w+_max$")
    pattern_sum = re.compile("^sum_\w+_sum$")
    pattern_max_cols = [col for col in aggregated.columns if pattern_max.match(col)]
    pattern_sum_cols = [col for col in aggregated.columns if pattern_sum.match(col)]
    aggregated['P_max'] = aggregated[pattern_max_cols].sum(axis=1)
    aggregated['P_sum'] = aggregated[pattern_sum_cols].sum(axis=1)
    aggregated.drop(columns=pattern_max_cols + pattern_sum_cols, inplace=True)

    # strip "count_" from columns starting with count
    aggregated.rename(columns={col: col[6:] for col in aggregated.columns if col.startswith('count')}, inplace=True)
    return aggregated

def sequences(df):
    df1 = df.drop(columns=['FLAG', 'PATTERN', 'OPEN', 'CREATE', 'DELETE', 'ENCRYPT', 'FILENAME'])

    # count the number of event type sequences (length 3)
    df1['NEXT'] = df1.groupby(['PID'])['TYPE'].transform(lambda col: col.shift(-1, fill_value='X'))
    df1['AFTER'] = df1.groupby(['PID'])['TYPE'].transform(lambda col: col.shift(-2, fill_value='X'))
    df1['SEQUENCE'] = df1[['TYPE', 'NEXT', 'AFTER']].apply(lambda row: ''.join(row.values.astype(str)), axis=1)

    aggregated = df1.groupby(['PID', 'SEQUENCE'])['TS'].agg('count').unstack(level='SEQUENCE', fill_value=0)

    # drop dummy sequences (containing X)
    aggregated.drop(columns=[col for col in aggregated.columns if 'X' in col], inplace=True)
    return aggregated


def prepare_dfs(df):
    df['TYPE'].replace([0,1,2,3], TYPE_NAMES, inplace=True)
    #df['TYPE'].replace(['0','1','2','3'], TYPE_NAMES, inplace=True)
    c,s = counts(df), sequences(df)
    combined = pd.concat([c, s], axis=1)
    combined = combined.reset_index()
    print(combined.dtypes)
    return combined

def preprocess_input(df: pd.DataFrame) -> np.ndarray:
    df.drop(columns=['PID'], inplace=True)
    training_features = feature_names
    df.drop(columns=[f for f in df.columns if f not in training_features], inplace=True)
    features = df.columns
    for f in training_features:
        if f not in features:
            df[f] = 0
    print("Columns for the scaler=", df.columns)
    print("Training Features=", training_features)
    df = df[training_features]
    print(df.head())
    scaler.transform(df)
    return df


def fetch_recent_events():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM events WHERE TS >= strftime('%s' , 'now', '-10 minutes');"
    events = pd.read_sql(query, conn)
    
    if events.empty:
        conn.close()
        return pd.DataFrame()
    
    events = prepare_dfs(events)
    pidarr = events['PID']
    events = preprocess_input(events)
    
    predictions = model.predict(events)
    prediction_df = pd.DataFrame({'PID': pidarr, 'Pred': predictions})
    
    # Insert predictions with value 1.0 into the incident table
    cursor = conn.cursor()
    for _, row in prediction_df.iterrows():
        if row['Pred'] == 1.0:
            cursor.execute(
                """
                INSERT OR REPLACE INTO incident (PID, TS)
                SELECT ?, strftime('%s', 'now')
                WHERE NOT EXISTS (
                    SELECT PID, TS FROM incident
                    WHERE PID = ?
                    AND TS >= strftime('%s', 'now', '-10 minutes')
                );
                """,
                (row['PID'], row['PID'])
            )
    
    conn.commit()
    conn.close()
    
    return prediction_df

def poll_database(interval=10):  # Poll every 10 seconds
    while True:
        recent_events = fetch_recent_events()
        if recent_events.size>0:
            print("Recent Events:\n", recent_events)  # Replace with your processing logic
        time.sleep(interval)

if __name__ == "__main__":
    poll_database()
