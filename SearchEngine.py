import txtai
import numpy as np
import pandas as pd
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

np.random.seed(1)

df = pd.read_csv('googleplaystore.csv')
titles =df.dropna().App.values

embeddings = txtai.Embeddings({
    'path':'sentence-transformers/all-MiniLM-L6-v2'
})


# embeddings.load('embeddings.tar.gz')
embeddings.index(titles)
embeddings.save('embeddings.tar.gz')

result = embeddings.search('workout',5)
print(result);

actual_results = [titles[x[0]] for x in result]
