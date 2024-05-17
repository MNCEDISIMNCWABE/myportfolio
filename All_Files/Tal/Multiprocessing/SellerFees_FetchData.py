from tqdm import tqdm
import pandas as pd
import sqlalchemy as db
from sqlalchemy import create_engine
import os
import calendar
from datetime import datetime
import pickle
import time


def Get_SellerIds():
    
    """
    @desc: this function pulls seller information.
    
    @params:
        - No parameters 
    
    @returns:
        - 
    """

    TAKE2_DB_USER = os.environ.get('TAKE2_DB_USER')
    TAKE2_DB_PASSWORD = os.environ.get('TAKE2_DB_PASSWORD')   
 
    MySQL_engine = db.create_engine('mysql+mysqlconnector://'+TAKE2_DB_USER+':'+TAKE2_DB_PASSWORD+'@cpt-hq-db02-b55.hq.takealot.com:3306/take2').connect()

    query=f'''select distinct(idSeller)
                from take2.sellers
    '''
    
    print(query)
    start_time = time.time()
    df_Sellers = pd.read_sql_query(query, MySQL_engine)
    SellersIDs = tuple(sorted(set(df_Sellers.idSeller))) 
    print("--- %s seconds ---" % (time.time() - start_time), 'Seller IDs')
    with open('SellersIDs.pickle', 'wb') as output:
        pickle.dump(SellersIDs, output)
    return


def Get_SellerFees(Month):

    with open('SellersIDs.pickle', 'rb') as data:
        idsellers = pickle.load(data)

    dt = datetime.strptime(Month, "%B %Y")
    day = dt.day
    month = dt.month
    nextmonth = month+1 if month <12 else 1
    year = dt.year
    nextyear = year if month < 12 else year+1
    chunk=400
    Transaction_Type=(1, 2, 3, 4, 5, 6, 7, 20, 22, 24, 25)
    SellerFees = []
    start_time = time.time()
    MySQL_engine = db.create_engine('mysql+mysqlconnector://knime:chi1Gojiek@Data-lakehouse-mysql-57-services12.hq.takealot.com:3306/seller_transaction_log').connect()
    for i in tqdm(range(0, len(idsellers), chunk)):
        for i in range(0, len(Transaction_Type), 1):
            query=f'''SELECT transaction_log.date_added, transaction_log.seller_id, transaction_log.value_inc_vat, 
                        transaction_log.vat_multiplier, type
                        FROM seller_transaction_log.transaction_reference
                        JOIN seller_transaction_log.transaction_log using (transaction_log_id)
                        JOIN seller_transaction_log.transaction_type using (transaction_type_id)
                        JOIN seller_transaction_log.type_group using (type_group_id)
                        WHERE reference_type_id = 1
                        AND transaction_log.seller_id IN {idsellers[i:i + chunk]}
                        AND transaction_log.date_added >= '{str(year)+'-'+str(month)+'-'+str(day)}' 
                        AND transaction_log.date_added < '{str(nextyear)+'-'+str(nextmonth)+'-'+str(day)}'
                        AND transaction_log.transaction_type_id = {Transaction_Type[i]};
                        '''
            df_SellerFees = pd.read_sql_query(query, MySQL_engine)
            SellerFees.append(df_SellerFees)   
    SellerFees_Data=pd.concat(SellerFees)
    with open(Month+'\SellerFees_Data.pickle', 'wb') as output:
        pickle.dump(SellerFees_Data, output)
    
    print("--- %s seconds ---" % (time.time() - start_time), 'Seller Fees '+Month)

    return 