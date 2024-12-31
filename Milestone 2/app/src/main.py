from cleaning import clean
from db import save_to_db
import pandas as pd
import os
import time
from run_producer import start_producer, stop_container
from consumer import consume_data

# Importing the data


def import_data(file_path):
    return pd.read_csv(file_path)


# Saving the data
def save_data(df, LookupTable):
    LookupTable.to_csv(
        'data/lookup_fintech_data_MET_P1_52_16824.csv', index=False)
    df.to_parquet(
        'data/fintech_data_MET_P1_52_16824_clean.parquet', index='pyarrow')


def main():
    if (os.path.exists('data/fintech_data_MET_P1_52_16824_clean.parquet') and os.path.exists('data/lookup_fintech_data_MET_P1_52_16824.csv')):
        print("Data already cleaned")
        cleaned_df = pd.read_parquet(
            'data/fintech_data_MET_P1_52_16824_clean.parquet')
        LookupTable = pd.read_csv(
            'data/lookup_fintech_data_MET_P1_52_16824.csv')
    else:
        print("Cleaning data")
        fintech_df = import_data('data/fintech_data_49_52_16824.csv')
        cleaned_df, LookupTable = clean(fintech_df)
        save_data(cleaned_df, LookupTable)
        print("Data cleaned and saved locally")

    save_to_db(cleaned_df, 'fintech_data_MET_P1_52_16824')
    save_to_db(LookupTable, 'lookup_fintech_data_MET_P1_52_16824')
    print("Data saved to database")

    print('Starting producer', flush=True)
    id = start_producer('52_16824', 'kafka:29092', 'fintech_topic')
    print('Producer started with id:', id)

    print('Consuming data', flush=True)
    consume_data('fintech_topic', 'kafka:29092')
    print('Data consumed')

    # try:
    print('Stopping producer')
    stop_container(id)
    print('Producer stopped')
    # except Exception as e:
    #     print('Failed to stop producer', e)


if __name__ == '__main__':
    main()
