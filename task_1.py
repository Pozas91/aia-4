from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
import numpy as np

########################################################################################################################
t0 = time()

# Categories to load
categories = [
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'sci.space'
]

print('')
print('Cargando el dataset de 20 newsgroups para las siguientes categorias:')
print(categories)

# Only get newsgroups from categories on above and only headers
dataset = fetch_20newsgroups(
    subset='all', categories=categories, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes')
)

# Get the size of all labels
labels = dataset.target
unique_labels = np.unique(labels).shape[0]

print('hecho en %fs' % (time() - t0))
print('')

########################################################################################################################

t0 = time()
print('Vectorizamos el conjunto de palabras')

# Vectorizer
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english', use_idf=True)
x = vectorizer.fit_transform(dataset.data)

print('hecho en %fs' % (time() - t0))
print('Número de ejemplos: %d, Número de características: %d' % x.shape)
print('')

########################################################################################################################

t0 = time()
print('Realizamos el algoritmo K-media, para organizar por clústeres')

# KMeans
k_means = KMeans(n_clusters=unique_labels)
k_means.fit(x)

print('hecho en %fs' % (time() - t0))
print('')