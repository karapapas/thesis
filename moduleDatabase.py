import re
import pandas as pd
import mysql.connector


class DatabaseMethods:

    @staticmethod
    def connect():
        return mysql.connector.connect(user="root",
                                       password="toor",
                                       host="127.0.0.1",
                                       database="mci_db")

    @staticmethod
    def fetch(select_clause):
        cnx = DatabaseMethods.connect()
        try:
            df = pd.read_sql(select_clause, cnx)
        except df.empty:
            print('No data to fetch for the query:', select_clause)
        except Exception as e:
            print('Error while fetching data:', e)
        cnx.close()
        return df

    @staticmethod
    def separate_target_class(df, target_class):
        df = df.rename(columns={target_class: "target_class"})
        for i in df.columns.array:
            if re.search('(^mmse_|^moca_)', i):
                df = df.drop([i], axis=1)
        df = df.set_index('gsId', drop=False)
        return df

    @staticmethod
    def keep_only_binned_target_classes(df):
        for i in df.columns.array:
            if not re.search('(^mmse_|^moca_)', i):
                df = df.drop([i], axis=1)
            elif re.search('(_diff$|_init$)', i):
                df = df.drop([i], axis=1)
        return df
