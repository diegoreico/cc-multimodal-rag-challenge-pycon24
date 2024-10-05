import pandas as pd

data = pd.read_csv('updated_data.csv')

print(data)
print(data.columns)

print(data['reviewed claim'][10])

print(data['unverified claim'][10])

print(data['meta_description'][10])

print(data['cr_country'][10])

print(data['meta_lang'][10])

print(data['cm_authors'][10])

print(data['cr_author_name'][10])

print(data['cr_item_reviewed_text'][10])

print(data['dataset'][10])

print(data['similarity'][10])

data = data.rename(columns={'cr_country': 'country', 'cr_author_name':'author name','meta_lang':'lang'})

data = data[['reviewed claim','unverified claim','similarity','country','lang','author name','cr_item_reviewed_text','cr_image','local_image_path']]

data.to_csv('./transformed.csv')

for i in range(0, len(data)):
    data.iloc[i:i+1,:].to_json(f'./single_documents/{i}.json',orient='records', lines=True)


