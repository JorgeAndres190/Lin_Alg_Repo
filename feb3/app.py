import pandas as pd 
import torch

df = pd.read_csv('data.csv')
X = torch.tensor(df.drop('Y', axis=1).to_numpy()).float()
Y = torch.tensor(df['Y'].to_numpy()).float().reshape(-1,1)

w = torch.tensor([
    [-1],
    [-2],
    [2],
    [1]
]).float()


b = torch.tensor([
    [-1]
])

Yhat = X@w + b
r = Yhat - Y
SSE = r.T@r
loss = SSE/15

print(loss)