import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


for _year in [2016,2017,2018]:
    zone, year, param = 'SE', str(_year), 'hu'
    fname = "../dataset/"+zone+"/ground_stations/"+zone+year+".csv"
    df = pd.read_csv(fname,parse_dates=[4])

    print(len(df))

    data_bron = df[df["number_sta"]==69029001]

    # Export data to CSV
    data_bron.to_csv(f"../dataset/extract/{zone}/ground_stations/{zone}{year}.csv", index=False)

