from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize

from time import time
import numpy as np

########################################################################################################################

tt = time()

print('')

########################################################################################################################

t0 = time()

# Categorias a cargar
categories = [
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'sci.space'
]

print('Cargando el dataset de 20 newsgroups para las siguientes categorias:')
print(categories)

# Cogemos todos los grupos de post, de las categorias dadas, de forma aleatoria eliminando cabeces, pies y citas.
dataset = fetch_20newsgroups(
    subset='all', categories=categories, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes')
)

# Obtenemos el número total de etiquetas diferentes
labels = dataset.target
unique_labels = np.unique(labels).shape[0]

print('Terminado en %fs' % (time() - t0))
print()

########################################################################################################################

t0 = time()

print('Realizamos el stemming de los posts del datasets')

stemmer = SnowballStemmer('english')

for i, data in enumerate(dataset.data):
    dataset.data[i] = ' '.join([stemmer.stem(token) for token in word_tokenize(data)])

print('Terminado en %fs' % (time() - t0))
print()

########################################################################################################################

t0 = time()
print('Vectorizamos el conjunto de palabras')

# Valor decimal [0.0, 1.0], cuando se construya el vocabulario se ignorarán los terminos que tengan mayor frecuencia
# documental a este umbral (Usado mayormente para evitar las stop-words, o palabras que aparezcan demasiado y no sean
# relevantes). Cuando se introduce un número entero referencia a la cantidad de palabras y no a la frecuencia
# documental.
max_df = 0.5

# Valor decimal [0.0, 1.0], cuando se construya se ignorarán los términos que tengan menor frecuencia umbral a este
# umbral (Usado mayormente para evitar palabras muy poco usadas, o demasiado específicas del documento). Cuando se
# introduce un número entero referencia a la cantidad de palabras y no a la frecuencia documental
min_df = 2

# Idioma de las stop-words
stop_words = 'english'

# Activa la frencuencia de documentación inversa en los pesos de las palabras.
use_idf = True

# Creamos un vector tfidf con las siguientes características
vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words, use_idf=use_idf)

# Entrenamos el vector y transformamos los datos en un vector numérico
x = vectorizer.fit_transform(dataset.data)

print('Terminado en %fs' % (time() - t0))
print('Nº de ejemplos: %d, Nº de características: %d' % x.shape)
print()

########################################################################################################################

t0 = time()
test = ['I\'m working with Windows for the IA subject.']
print('Vectorizamos el conjunto de prueba')

# Pasamos el conjunto de test a un vector numérico con la información previa entrenada, de esta forma tendremos las
# posiciones de las palabras de la frase dada
x_test = vectorizer.transform(test)

print('Terminado en %fs' % (time() - t0))
print('Nº de ejemplos: %d, Nº de características: %d' % x_test.shape)
print()

########################################################################################################################

t0 = time()
print('Realizamos el algoritmo K-media, para organizar por clústeres')

# Ahora realizamos el algoritmo K-medias para optimizar la ejecución del problema y dar un conjunto de datos donde se
# clasifique nuestro conjunto de entrenamiento
k_means = KMeans(n_clusters=unique_labels)
k_means.fit(x)

print('Terminado en %fs' % (time() - t0))
print('Nº de centros: %d, Nº de características: %d' % k_means.cluster_centers_.shape)
print()

########################################################################################################################

t0 = time()
print('Clasificamos el centro donde quedaría nuestro conjunto de prueba')

# Sacamos el centro del conjunto de prueba
prediction = k_means.predict(x_test)

print('Terminado en %fs' % (time() - t0))
print('El conjunto de prueba ha quedado clasificado en los centros: %s' % prediction)
print()

########################################################################################################################

t0 = time()
print('Sacamos los 10 "mejores" términos los clústers predichos: %s' % prediction)

# Ordenamos los centros y nos quedamos con el predicho
order_centroid = k_means.cluster_centers_.argsort()[:, ::-1]
total_centers = len(order_centroid)

# Obtenemos todas las palabras vectorizadas
terms = vectorizer.get_feature_names()

for i in prediction:
    print('Cluster %d:' % i)
    for j in order_centroid[i, :10]:
        print(' %s' % terms[j], end='')
    print()

print('Terminado en %fs' % (time() - t0))
print()

########################################################################################################################

print('La ejecución total del programa ha sido de %fs' % (time() - tt))
