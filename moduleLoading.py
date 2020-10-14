import re
import pandas as pd
import mysql.connector

class LoadingMethods:
    
    # def __init__(self, testVar):
    #     self.testVar = testVar

    def connectAndFetch(self, dbHost, dbName, dbUser, dbPass, selectClause):
        cnx = mysql.connector.connect(user=dbUser, password=dbPass, host=dbHost, database=dbName)
        dataset = pd.read_sql(selectClause, cnx)
        cnx.close()
        return dataset
    
    def separateTargetClass(self, df, targetClass):
        df = df.rename(columns={targetClass:"target_class"})
        for i in df.columns.array: 
            if(re.search('(^mmse_|^moca_)', i)):
                df = df.drop([i], axis=1)
        df = df.set_index('gsId', drop = False)
        self.df=df
        return df
