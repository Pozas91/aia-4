from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from time import time
from operator import itemgetter

import numpy as np
import utils.text as ut


def task_1(texts_samples: list, texts_showed: int):

    """
    Método principal para ejecutar la tarea 1
    :param texts_samples:
        Textos para de los cuales queremos saber la cercanía
    :param texts_showed:
        Número de textos más afines a mostrar para los textos dados por el parámetro
    :return:
    """

    # Comprobación de que los parámetros indicados deben ser los correctos
    if not texts_samples:
        raise ValueError('Debe haber al menos un texto de ejemplo.')

    if texts_showed <= 0:
        raise ValueError('Se debe mostrar al menos un texto similar por cada ejemplo dado.')

    ####################################################################################################################

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

    print('Cargando el dataset de twenty_newsgroups para las siguientes categorias... {0}'.format(categories))

    # Cogemos todos los grupos de post, de las categorias dadas, de forma aleatoria eliminando cabeces, pies y citas.
    dataset = fetch_20newsgroups(
        subset='all', categories=categories, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes')
    )

    print('Terminado en {0:.2f}s'.format(time() - t0))
    print()

    ####################################################################################################################

    t0 = time()
    print('Vectorizamos el conjunto de palabras...', end=' ')

    # Valor decimal [0.0, 1.0], cuando se construya el vocabulario se ignorarán los terminos que tengan mayor
    # frecuencia documental a este umbral (Usado mayormente para evitar las stop-words, o palabras que aparezcan
    # demasiado y no sean relevantes). Cuando se introduce un número entero referencia a la cantidad de palabras y no
    #  a la frecuencia documental.
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
        max_df=max_df, min_df=min_df, stop_words=stop_words, use_idf=use_idf, tokenizer=ut.stemmer_tokenizer
    )

    # Entrenamos el vector y transformamos los datos en un vector numérico
    x_train = vectorizer.fit_transform(dataset.data)

    print('terminado en {0:.2f}s'.format(time() - t0))
    print('Nº de ejemplos: {0:d}, Nº de características: {1:d}'.format(x_train.shape[0], x_train.shape[1]))
    print()

    ####################################################################################################################

    t0 = time()
    print('Vectorizamos el conjunto de prueba...', end=' ')

    # Pasamos el conjunto de test a un vector numérico con la información previa entrenada, de esta forma tendremos las
    # posiciones de las palabras de la frase dada
    x_test = vectorizer.transform(texts_samples)

    print('terminado en {0:.2f}s'.format(time() - t0))
    print('Nº de ejemplos: {0:d}, Nº de características: {1:d}'.format(x_test.shape[0], x_test.shape[1]))
    print()

    ####################################################################################################################

    t0 = time()
    print('Realizamos el algoritmo K-media, para organizar por clústeres...', end=' ')

    # Ahora realizamos el algoritmo K-medias para optimizar la ejecución del problema y dar un conjunto de datos donde se
    # clasifique nuestro conjunto de entrenamiento. Se han probado diferentes tamaños de clústeres:
    # 10 --> 141s
    # 20 --> 116s
    # 30 --> 149s
    # 40 --> 128s
    # 50 --> 343s
    # dando como mejor resultado y más rápido 20 clústeres.
    k_means = KMeans(n_clusters=20)
    k_means.fit(x_train)

    print('terminado en {0:.2f}s'.format(time() - t0))
    print('Nº de ejemplos: {0:d}, Nº de características: {1:d}'.format(
        k_means.cluster_centers_.shape[0], k_means.cluster_centers_.shape[1]
    ))
    print()

    ####################################################################################################################

    t0 = time()
    print('Clasificamos el centro donde quedaría nuestro conjunto de prueba...', end=' ')

    # Sacamos el centro del conjunto de prueba
    prediction = k_means.predict(x_test)

    print('terminado en {0:.2f}s'.format(time() - t0))
    print('El conjunto de prueba ha quedado clasificado en los centros: {0}'.format(prediction))
    print()

    ####################################################################################################################

    t0 = time()

    print('Sacamos los {0:d} posts más parecidos a los mencionados por el usuario.'.format(texts_showed))

    posts = dict()

    for center_index, center in enumerate(prediction):
        for label_index in np.where(k_means.labels_ == center)[0]:

            # Obtenemos la similitud entre la frase del conjunto de prueba, y la del conjunto de entrenamiento
            sim = ut.similarity_words(x_test[center_index].toarray()[0], x_train[label_index].toarray()[0])

            if center_index in posts:
                values = posts.get(center_index)
                values.append((label_index, sim))
                posts.update({center_index: values})
            else:
                posts.update({center_index: [(label_index, sim)]})

    # Por cada texto de prueba
    for center in posts:

        # Sacamos el cluster al que pertenece
        cluster = prediction[center]

        # Sacamos las palabras ordenadas de mayor a menor de forma eficiente
        labels = sorted(posts.get(center), key=itemgetter(1), reverse=True)

        print('Para el texto de prueba: ')
        print(texts_samples[center])
        print()

        print('Se han encontrado las siguientes textos más similares:')

        # Nos quedamos con las x primeros y los mostramos
        for key, sim in labels[:texts_showed]:

            print('\n {0}'.format(dataset.data[key]))

            # Añadimos una línea separatoria entre los documentos
            print('\n' + '-' * 50)

    print('Terminado en {0:.2f}s'.format(time() - t0))
    print()


if __name__ == '__main__':
    print('Iniciando la ejecución del programa principal...', end='\n\n')

    t0 = time()

    samples = [
        'I\'m working with Windows for the IA subject.',
        'New nvidia GTX2048 in shops, for only 10000$'
    ]

    task_1(texts_samples=samples, texts_showed=3)

    print('La ejecución total del programa ha sido de {0:.2f}s'.format(time() - t0))
