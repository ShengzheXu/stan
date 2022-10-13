from stannetflow import STANSynthesizer, STANCustomDataLoader
from stannetflow.artificial.datamaker import artificial_data_generator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import numpy as np
import matplotlib.pyplot as plt


# case 1. 3-4 columns from the credit card data, but independently sampled.. 
df = pd.read_csv('./test_cat.csv')[:100]
scaler = MinMaxScaler()
df[['Amount_normed']] = scaler.fit_transform(df[['Amount']])

disc_encoder = OrdinalEncoder()
df[['Merchant City_normed', 'Is Fraud?_normed']] = disc_encoder.fit_transform(df[['Merchant City', 'Is Fraud?']])

print(df[['Amount_normed', 'Merchant City_normed', 'Is Fraud?_normed']])
print(disc_encoder.categories_)

#
# Usecase 1: whole batch training - stan.fit()
# 

df = df[['Amount_normed', 'Merchant City_normed', 'Is Fraud?_normed']]
adg = artificial_data_generator(weight_list=None)
adg.df_naive = df

window_size = 1
column_size = 3
categorical_columns = {1:5, 2:1}
stan = STANSynthesizer(dim_in=column_size, dim_window=window_size, categorical_columns=categorical_columns)

X, y = adg.agg(agg=window_size)
stan.fit(X, y, epochs=1000)

samples = stan.sample(100)
print(samples)


#
# Usecase 2: batch training - stan.batch_fit()
# 

window_size = 1
column_size = 3
adg = artificial_data_generator(weight_list=None)
adg.df_naive = df
X, y = adg.agg(agg=window_size)

pd.DataFrame(X.view(X.size(0), -1).numpy()).to_csv('test_demo.csv', index=False, header=None)
train_loader = STANCustomDataLoader(csv_path='./test_demo.csv', height=window_size, width=column_size).get_loader()

categorical_columns = {1:5, 2:1}
stan = STANSynthesizer(dim_in=column_size, dim_window=window_size, categorical_columns=categorical_columns)
stan.batch_fit(train_loader, epochs=100)

samples = stan.sample(100)
print(samples)
