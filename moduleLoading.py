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
    
    def separateTargetClass(self, dataset, targetClass):
        dataset = dataset.rename(columns={targetClass:"target_class"})
        for i in dataset.columns.array: 
            if(re.search('(^mmse_|^moca_)', i)):
                dataset = dataset.drop([i], axis=1)
        return dataset
