import pandas as pd
import turicreate as tc

data_dummy = pd.read_csv("new_dummy.csv")

user_id = 'Garage'
item_id = 'product_name'
n_rec = 3

final_model = tc.item_similarity_recommender.create(tc.SFrame(data_dummy),
                                            user_id=user_id, 
                                            item_id=item_id,
                                            target='purchase_dummy', similarity_type='cosine')

def create_output(model, user_to_recommend, n_rec):
    recomendation = model.recommend(users=user_to_recommend, k=n_rec)
    return recomendation

user_to_recommend_item =["abcd"]
df_output = create_output(final_model,user_to_recommend_item , n_rec)

print(df_output["product_name"])