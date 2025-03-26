import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

try:
    in_file = "./data/" + sys.argv[1] + ".csv"
    file = open(in_file, "r")
except:
    sys.exit("ERROR. Bad file name")
df = pd.read_csv(file)
print("SHAPE : ", df.shape)
print("\n\n\n\nINFO : ")
print(df.info())
print("\n\n\n\nDESCRIBE : ")
print(df.describe())
print("\n\n\n\nHEAD : ")
print(df.head())
print("\n\n\n\nNAN : ")
print(df.isna().sum())
print("\n\n\n\nNULL : ")
print(df.isnull().sum())
print("\n\n\n\nDUPLICATES : ")
print(df.duplicated().sum())
print("\n\n\n\nLigne à 3 NAN (hc,nox,hcnox) : ", df[df[['hc', 'nox', 'hcnox']].isnull().all(axis=1)].shape[0]," lignes \n")
print(df[df[['hc', 'nox', 'hcnox']].isnull().all(axis=1)])
print(df[df['id'] == 769])
print(df[df['id'] == 796])

cols = df.columns
fig = plt.figure(figsize = (20,20))
# pair plot
g = sns.pairplot(data=df[cols])

path = "./plots/" + sys.argv[1] + "_pairplot.pdf"

g.savefig(path, format='pdf')