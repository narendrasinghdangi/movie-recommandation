import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv("new_data.csv")

while 1:

    print("")
    purchase_id= int(input("Enter purcahse id:- "))
    make  = input("Enter Make:- ")
    model = input("Enter model:- ")
    product_name = input("Enter product Name:- ")

    prod = str(make)+"_"+str(model)+"_"+str(product_name)

    data.loc[len(data.index)] = [purchase_id,prod, 1]

    data_pivot = data.pivot(index = 'product', columns = 'PurchaseOrderId', values = 'purchase_count').fillna(0)
    data_matrix = csr_matrix(data_pivot.values)

    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(data_matrix)

    query_index = data_pivot.shape[0]
    distances, indices = model_knn.kneighbors(data_pivot.iloc[query_index,:].values.reshape(1, -1))


    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for:- {0}:\n'.format(data_pivot.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, data_pivot.index[indices.flatten()[i]], distances.flatten()[i]))