import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import csv
import requests

key = '59N6YQYDT1BDEQFS'

def import_stocks_tickers(date='today', tickers='All', listing_status='All', exchange='All', asset_type='All'):
    """
    Returns a table containing stocks listings information.
    
    - date: 'YYYY-MM-DD' format (string) - today by default
    - listing_status: 'active', 'delisted', or 'All' by default (string)
    - exchange: 'NYSE', 'NYSE ARCA', 'BATS', 'NASDAQ', 'NYSE MKT' (list) - All by default
    - asset_type: 'Stock', 'ETF' (list) - All by default
    - tickers: (list)
    
    
    """
    
    import csv
    import requests
    
    base_url = 'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={}'.format(key)
    
    if date != 'today':
        base_url = base_url + '&date={}'.format(date)
        
    
    if listing_status == 'active':
        base_url1 = base_url + '&state=active'
    elif listing_status == 'delisted':
        base_url1 = base_url + '&state=delisted'
    elif listing_status == 'All':
        base_url1 = base_url + '&state=active'
        base_url2 = base_url + '&state=delisted'
    else:
        raise ValueError("Invalid value for 'listing_status'. Accepted values are 'active', 'delisted' and 'All'")

        
    with requests.Session() as s:
        download = s.get(base_url1)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        tickers_data = list(cr)
        symbols = pd.DataFrame(tickers_data[1:], columns=tickers_data[0])
    
    if listing_status == 'All':
        download2 = s.get(base_url2)
        decoded_content2 = download2.content.decode('utf-8')
        cr2 = csv.reader(decoded_content2.splitlines(), delimiter=',')
        tickers_data2 = list(cr2)
        symbols2 = pd.DataFrame(tickers_data2[1:], columns=tickers_data2[0])
        
        symbols = pd.concat([symbols, symbols2], axis=0)
        symbols.reset_index(inplace=True)
        symbols=symbols.drop(columns='index', axis=1)
    
    if tickers != 'All':
        symbols = symbols.loc[symbols.symbol.isin(tickers)]
    
    if exchange != 'All':
        symbols = symbols.loc[symbols.exchange.isin(exchange)]
    
    if asset_type != 'All':
        symbols = symbols.loc[symbols.assetType.isin(asset_type)]
    
    return symbols


def import_sql_data(table, database='securities_master'):
    from sqlalchemy import create_engine
    import pymysql

    conn = pymysql.connect(
    host='localhost',
    user='sec_user',
    password='xT_Wrestling09',
    database=database,
    )
    
    # Create a cursor object
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    # Execute a query
    cursor.execute("SELECT * FROM {}".format(table))

    # Fetch all the rows
    rows = cursor.fetchall()

    # Convert to pandas DataFrame
    df = pd.DataFrame(rows)
    
    return df

def import_stock_data(ticker, table = "sp500_daily_adj", database='securities_master', start_date=None, end_date=None):
    from sqlalchemy import create_engine
    import pymysql

    conn = pymysql.connect(
    host='localhost',
    user='sec_user',
    password='xT_Wrestling09',
    database=database,
    )
    
    # Create a cursor object
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    
    # Prepare the SQL query with placeholders for ticker and date range
    query = "SELECT * FROM {} WHERE ticker = %s".format(table)
    parameters = [ticker]

    if start_date:
        query += " AND date >= %s"
        parameters.append(start_date)

    if end_date:
        query += " AND date <= %s"
        parameters.append(end_date)

    # Execute a query
    cursor.execute(query, parameters)

    # Fetch all the rows
    rows = cursor.fetchall()

    # Convert to pandas DataFrame
    df = pd.DataFrame(rows)
    
    df = df.set_index("date")
    df['price'] = df['5. adjusted close']
    df = df[['ticker', 'price']]
    df.sort_index(inplace=True)
    
    return df

def create_sql_table(df, key, table_name, database, create=None, update=None):
    import pymysql
    from sqlalchemy import create_engine
    from datetime import datetime
    
    # Define default behavior
    if create is None and update is None:
        create = True
        update = False

    # Define behavior when only 'update' is specified
    if update and create is None:
        create = False

    # Define behavior when only 'create' is specified
    if create and update is None:
        update = False
        
    if create == update:
        raise ValueError("Only one of 'create' and 'update' should be True")

    conn = pymysql.connect(
        host='localhost',
        user='sec_user',
        password='xT_Wrestling09',
        database=database,
    )
    
    cursor = conn.cursor()
    
    SCHEMA = list(df.columns.values)
    SCHEMA = ',\n'.join([f"`{column}` VARCHAR(255)" for column in SCHEMA])
    key_constraint = f"PRIMARY KEY (`{key[0]}`)"
    
    cursor.execute(f"SHOW TABLES LIKE '{table_name}';")
    result = cursor.fetchone()

    if not result:
        create_table_query = """
        CREATE TABLE `{}` (
            {},
            {}
        );
        """.format(table_name, SCHEMA, key_constraint)

        cursor.execute(create_table_query)
    
    engine = create_engine('mysql+pymysql://sec_user:xT_Wrestling09@localhost:3306/{}'.format(database))
    
    df['updated_at'] = datetime.now()
    
    
    if create == True:
        df['created_at'] = datetime.now()  
        df['updated_at'] = datetime.now()  # Add this line 

        df.to_sql(table_name, con=engine, if_exists='replace', index=False)

        create_trigger_query = """
        CREATE TRIGGER update_timestamp
        BEFORE INSERT ON `{}` 
        FOR EACH ROW
        SET NEW.updated_at = NOW();
        """.format(table_name)

        cursor.execute(create_trigger_query)
    else:
        if update:
            # Read existing data
            existing_df = import_sql_data(table_name, database)

            # Find new data that's not in the existing data
            new_data = pd.concat([df, existing_df]).drop_duplicates(subset=key, keep=False)

            # Check if there's any new data to append
            if not new_data.empty:
                print("New data has been added.")
                df['created_at'] = datetime.now()  
                df['updated_at'] = datetime.now()  # Add this line 
                df.to_sql(table_name, con=engine, if_exists='replace', index=False)
                
                
                create_trigger_query = """
                CREATE TRIGGER update_timestamp
                BEFORE INSERT ON `{}` 
                FOR EACH ROW
                SET NEW.updated_at = NOW();
                """.format(table_name)
                
                cursor.execute(create_trigger_query)
                
                #new_data.to_sql(table_name, con=engine, if_exists='append', index=False)
                
            else:
                print("No new data to update.")
                return
        else:
            raise ValueError("Only one of 'create' and 'update' should be True")