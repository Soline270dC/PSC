import pandas as pd
import numpy as np

Q = np.array((3.3241, 5.1292, 6.4897, 7.1301))

df1 = pd.read_csv("data/1-station_49.csv")
df2 = pd.read_csv("data/2-station_80.csv")
df3 = pd.read_csv("data/3-station_40.csv")
df4 = pd.read_csv("data/4-station_63.csv")

df1["S"] = df1["W_13"] + df1["W_14"] + df1["W_14"]
df1 = df1[["S", "YIELD"]]
df1.rename(columns = {"S" : "S1", "YIELD" : "Y1"}, inplace = True)
df2["S"] = df2["W_13"] + df2["W_14"] + df2["W_14"]
df2 = df2[["S", "YIELD"]]
df2.rename(columns = {"S" : "S2", "YIELD" : "Y2"}, inplace = True)
df3["S"] = df3["W_13"] + df3["W_14"] + df3["W_14"]
df3 = df3[["S", "YIELD"]]
df3.rename(columns = {"S" : "S3", "YIELD" : "Y3"}, inplace = True)
df4["S"] = df4["W_13"] + df4["W_14"] + df4["W_14"]
df4 = df4[["S", "YIELD"]]
df4.rename(columns = {"S" : "S4", "YIELD" : "Y4"}, inplace = True)

df = pd.concat([df1, df2, df3, df4], axis = 1)
df = df[["Y1", "Y2", "Y3", "Y4"]][(df["S1"] <= Q[0]) & (df["S2"] <= Q[1]) & (df["S3"] <= Q[2]) & (df["S4"] <= Q[3])]
a = df.to_numpy()

np.save("data/round1", a)