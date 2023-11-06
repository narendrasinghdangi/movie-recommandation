import pandas as pd
import turicreate as tc

user_id = 'PurchaseOrderId'
item_name = "product"
n_rec = 3

data_dummy = pd.read_csv("new_data.csv")


while 1:

    print("")
    purchase_id= 5
    make  = input("Enter Make:- ")
    model = input("Enter model:- ")
    product_name = input("Enter product Name:- ")

    prod = str(make)+"_"+str(model)+"_"+str(product_name)

    data_dummy.loc[len(data_dummy.index)] = [purchase_id,prod, 1]


    final_model = tc.item_similarity_recommender.create(tc.SFrame(data_dummy),
                                        user_id=user_id,item_id=item_name,
                                        target='purchase_count', similarity_type='cosine',verbose=0)

    def create_output(model, user_to_recommend, n_rec):
        recomendation = model.recommend(users=user_to_recommend, k=n_rec)
        return recomendation

    user_to_recommend_item =[purchase_id]
    df_output = create_output(final_model,user_to_recommend_item , n_rec)


    out=df_output["product"]
    for i in range(n_rec):

        a= out[i].split("_")[0]
        b= out[i].split("_")[1]
        c= out[i].split("_")[2]

        print(f"Make:-{a}, and Model:-{b}, and Item :- {c}")