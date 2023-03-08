
# STEPS :

# 1. Preparing the Data

# 2. Creating the CLTV Data Structure

# 3. BGNBD, Establishment of Gamma-Gamma Models, Calculation of CLTV

# 4. Creating Segments by CLTV

####################################################

# 1. Preparing the Data

import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

df_ = pd.read_csv("./flo_data_20K.csv")
df = df_.copy()
df.head()
df.describe().T
df.describe([0.01,0.25,0.5,0.75,0.99]).T

def outlier_thresholds(dataframe, variable):
    '''
        Sets the boundaries of outliers.
    Parameters
    ----------
    dataframe :DataFrame
    variable : int

    Returns
    -------
    low_limit : The low limit of data
    up_limit : The up limit of data
    '''
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    '''
        The function that necessary to suppress outliers
    Parameters
    ----------
    dataframe: DataFrame
    variable : int

    Returns
    -------
    '''
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

columns = ["order_num_total_ever_online",
           "order_num_total_ever_offline",
           "customer_value_total_ever_offline",
           "customer_value_total_ever_online"
           ]

for col in columns:
    replace_with_thresholds(df, col)

df["order_num_total"] = df["order_num_total_ever_online"] + \
                        df["order_num_total_ever_offline"]

df["customer_value_total"] = df["customer_value_total_ever_offline"] + \
                             df["customer_value_total_ever_online"]

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# 2. Creating the CLTV Data Structure

df["last_order_date"].max()  # 2021-05-30
analysis_date = dt.datetime(2021, 6, 1)

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - 
                                   df["first_order_date"]).
                                  astype('timedelta64[D]')) / 7

cltv_df["T_weekly"] = ((analysis_date - 
                        df["first_order_date"]).
                       astype('timedelta64[D]')) / 7

cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

cltv_df.head()

# 3. BGNBD, Establishment of Gamma-Gamma Models, Calculation of CLTV

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

cltv_df.sort_values("exp_sales_3_month", ascending=False)[:10]

cltv_df.sort_values("exp_sales_6_month", ascending=False)[:10]


ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])
cltv_df.head()

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # aylık
                                   freq="W",  # verilen bilgiler aylık mı? haftalık mı?
                                   discount_rate=0.01)  # kampanya etkisi göz önünde bulunduruluyor.
cltv_df["cltv"] = cltv

cltv_df.head()

cltv_df.sort_values("cltv", ascending=False)[:20]

# 4. Creating Segments by CLTV

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

cltv_df.groupby("cltv_segment").agg({"count", "mean", "sum"})

cltv_df[["segment", "recency_cltv_weekly", "frequency", "monetary_cltv_avg"]]. \
    groupby("cltv_segment"). \
    agg( ["mean", "count"])
   

