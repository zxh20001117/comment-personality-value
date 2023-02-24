import pandas as pd
import os

location_list = [
    'London_Reviews'
    , 'new york_Reviews'
    , 'Shanghai_Reviews'
]

path = 'viewDatas'
filenames = os.listdir(path)

for i in location_list:
    pandas_list = []
    for j in filenames:
        if i in j:
            pandas_list.append(pd.read_excel(f"{path}/{j}"))
    pd.concat(pandas_list).to_excel(f'dataProcess/{i} merged.xlsx')
    print(f'{len([k for k in filenames if i in k])} completed for {i}')

all_list = []
for i in location_list:
    all_list.append(pd.read_excel(f'dataProcess/{i} merged.xlsx'))
pd.concat(all_list).to_excel('dataProcess/all merged.xlsx')
