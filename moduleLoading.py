# -*- coding: utf-8 -*-

import pandas as pd
import mysql.connector

class Loaders:
    
    def connectAndFetch(dbHost, dbName, dbUser, dbPass, selectClause):
        # cnx = mysql.connector.connect(user='root', password='toor', host='127.0.0.1', database='mci_db')
        cnx = mysql.connector.connect(user=dbUser, password=dbPass, host=dbHost, database=dbName)
        # dataset = pd.read_sql("SELECT * FROM v5_mmse_pre", cnx)
        dataset = pd.read_sql(selectClause, cnx)
        cnx.close()
        return dataset
    
