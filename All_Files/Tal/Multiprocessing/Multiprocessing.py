from tqdm import tqdm
import pandas as pd
import sqlalchemy as db
from sqlalchemy import create_engine
import mysql.connector
import psycopg2
import time
from datetime import datetime
import pickle

def Get_Views(Month):
        
    MySQL_engine = db.create_engine('mysql+mysqlconnector://keletso.dithebe:Shaibai9Shee@cpt-hq-db02-b55.hq.takealot.com:3306/take2').connect()
    
    dt = datetime.strptime(Month, "%B %Y")
    day = dt.day
    month = dt.month
    nextmonth = month+1 if month <12 else 1
    year = dt.year
    nextyear = year if month < 12 else year+1
    
    query=f'''select b.SellerType, c.Division, c.ReportingDepartment, sum(views)
                from retail.PageViews_2021 b
                inner join  bi_general.view_business_hierarchy c
                  on c.idProduct = b.idProduct
                where b.SOH_Buyable = 1
                and (Date >= '{str(year)+'-'+str(month)+'-'+str(day)}' 
                and Date < '{str(nextyear)+'-'+str(nextmonth)+'-'+str(day)}')
                group by 1,2,3;
                    '''
    print(query)
    start_time = time.time()
    df_Views = pd.read_sql_query(query, MySQL_engine)
    df_Views['Month'] = Month
    df_Views.to_csv(Month+'\Views.csv', mode='a')   
    print("--- %s seconds ---" % (time.time() - start_time), Month)
    return 
