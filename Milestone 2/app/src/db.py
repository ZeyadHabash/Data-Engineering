from sqlalchemy import create_engine

engine = create_engine('postgresql://root:root@pgdatabase:5432/testdb')


def save_to_db(cleaned, table_name):
    if (engine.connect()):
        print('Connected to Database')
        try:
            print('Writing cleaned dataset to database')
            cleaned.to_sql(table_name,
                           con=engine,
                           if_exists='fail')
            print('Done writing to database')
        except ValueError as vx:
            print('Cleaned Table ' + table_name + ' already exists.')
        except Exception as ex:
            print(ex)
    else:
        print('Failed to connect to Database')


def save_message_to_db(message, table_name='fintech_data_MET_P1_52_16824'):
    if (engine.connect()):
        print('Connected to Database')
        try:
            print('Writing message to database')
            message.to_sql(table_name,
                           con=engine,
                           if_exists='append',
                           index=True,
                           index_label='customer_id')
            print('Done writing message to database')
        except Exception as ex:
            print(ex)
    else:
        raise Exception('Could not connect to database while saving message')
