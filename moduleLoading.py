import re
import pandas as pd
import mysql.connector


class LoadingMethods:

    @staticmethod
    def connect_and_fetch(db_host, db_name, db_user, db_pass, select_clause):
        cnx = mysql.connector.connect(user=db_user, password=db_pass, host=db_host, database=db_name)
        df = pd.read_sql(select_clause, cnx)
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
