from pathlib import Path
import pandas as pd
import numpy as np
import os
from glob import glob
import csv
from multiprocessing import Pool

'''
def main():
    csv_path = "../input/features"
    csv_path_ary = glob.glob(csv_path)

    read_csv(csv_path_ary)

def read_csv(csv_path_ary):
    p = Pool(os.cpu_count())
    df = pd.concat(p.map(read_csv, csv_path_ary))
    p.close()

if __name__ == '__name__':
    main()
'''
path_to_feature_dir = "../input/features"
rootdir = Path(path_to_feature_dir)
files = list(rootdir.glob("**/*.csv"))
df = pd.DataFrame()
for file in files:
    dfs = pd.read_csv(file, delimiter=';', index_col=0)
    dfs = dfs.loc[dfs.iloc[:,0] != 0] # csv見たら、最後の方一部の特徴量が0が続いていたため、それらを除く。
    dfs = dfs.mean(axis=0).to_frame().T
    dfs.index = [file.stem]
    df = pd.concat([df,dfs], axis=0)
df.index = [int(i) for i in df.index]
df.index_name = 'song_id'
df.columns = ["OpenSMILE__{}".format(s) for s in df.columns.tolist()]

df.to_csv('../input/OpenSMILE.csv')
