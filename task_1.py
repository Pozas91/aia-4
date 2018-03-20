from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from time import time
from operator import itemgetter

import utils.text as ut

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

print('Cargando el dataset de twenty_newsgroups para las siguientes categorias:')
print(categories)

# Cogemos todos los grupos de post, de las categorias dadas, de forma aleatoria eliminando cabeces, pies y citas.
dataset = fetch_20newsgroups(
    subset='all', categories=categories, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes')
)

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
vectorizer = TfidfVectorizer(
    # max_df=max_df, min_df=min_df, stop_words=stop_words, use_idf=use_idf, tokenizer=ut.stemmer_tokenizer
    max_df=max_df, min_df=min_df, stop_words=stop_words, use_idf=use_idf
)

# Entrenamos el vector y transformamos los datos en un vector numérico
x_train = vectorizer.fit_transform(dataset.data)

print('Terminado en %fs' % (time() - t0))
print('Nº de ejemplos: %d, Nº de características: %d' % x_train.shape)
print()

########################################################################################################################

t0 = time()
test = ['I\'m working with Windows for the IA subject.', 'I\'m working with Windows for the IA subject.']
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
# clasifique nuestro conjunto de entrenamiento.
k_means = KMeans(n_clusters=50)
k_means.fit(x_train)

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
print('Sacamos los 10 posts más parecidos a los mencionados por el usuario.')

posts = dict()
for i, center in enumerate(prediction):
    for j, label in enumerate(k_means.labels_):

        if center == label:

            # Obtenemos la similitud entre la frase del conjunto de prueba, y la del conjunto de entrenamiento
            sim = ut.similarity_words(x_test[i].toarray()[0], x_train[j].toarray()[0])

            if i in posts:
                values = posts.get(i)
                values.append((j, sim))
                posts.update({i: values})
            else:
                posts.update({i: [(j, sim)]})

# Por cada texto de prueba
for center in posts:

    # Sacamos el cluster al que pertenece
    cluster = prediction[center]

    # Sacamos las palabras ordenadas de mayor a menor de forma eficiente
    labels = sorted(posts.get(center), key=itemgetter(1), reverse=True)

    print('Para el texto de prueba: ')
    print(test[center])
    print()

    print('Se han encontrado las siguientes textos más similares:')

    # Textos mostrados
    texts_showed = 3

    # Nos quedamos con las x primeros y los mostramos
    for key, sim in labels[:texts_showed]:
        print(dataset.data[key])
        print('-------------------------------------------------------------------------------------------------------')

print('Terminado en %fs' % (time() - t0))
print()

########################################################################################################################

print('La ejecución total del programa ha sido de %fs' % (time() - tt))
