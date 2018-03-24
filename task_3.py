from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import pandas as pd
import utils.text as ut
import os
from time import time


def task_3(reviews_samples: list, n_jobs: int, training_taken=200):

    """
    Método principal para ejecutar la tarea 3
    :param reviews_samples:
        Opiniones de ejemplo que queremos clasificar de acuerdo al sentimiento que transmiten
    :param n_jobs:
        Nº de procesadores usados para ejecutar el problema
    :param training_taken:
        Nº de datos de entrenamiento cogidos para entrenar al modelo
    :return:
    """

    # Comprobación de que los parámetros indicados deben ser los correctos
    if not reviews_samples:
        raise ValueError('Debe haber al menos una opinión de ejemplo.')

    if n_jobs <= 0 and n_jobs is not -1:
        raise ValueError('El argumento n_jobs debe ser o -1 para trabajar con todos los núcleos del sistema, '
                         'o el nº de núcleos indicado')

    # Si el número indicado es menor a 0
    # (Si es mayor al número total de películas que tenemos, simplemente cogerá las máximas posibles)
    if training_taken <= 0:
        raise ValueError('El argumento de valores de entrenamiento tomados debe ser un número mayor a 1.')

    ####################################################################################################################

    t0 = time()

    print('Cargando y pasando datos desde el csv...', end=' ')

    # Cargamos los datos con panda desde el CSV
    movie_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/movie_data.csv'))

    x = list()
    y = list()

    # Indicamos los tipos de sentimientos que tenemos
    target_names = ['Negativo', 'Positivo']

    # Cogemos solo el número de datos indicado en la cabecera del método
    for values in movie_data.values[:training_taken]:
        review, sentiment = values
        x.append(review)
        y.append(sentiment)

    print('terminado en {0:.2f}s'.format(time() - t0))
    print()

    ####################################################################################################################

    t0 = time()

    print('Preparando datos para el ejemplo...', end=' ')

    x_train, y_train, x_sample = x, y, reviews_samples

    print('terminado en {0:.2f}s'.format(time() - t0))
    print()

    ####################################################################################################################

    t0 = time()

    print('Creamos un Pipeline con el vectorizador y el clasificador...', end=' ')

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=ut.stemmer_tokenizer, stop_words='english')),
        ('multinomial', MultinomialNB())
    ])

    print('terminado en {0:.2f}s'.format(time() - t0))
    print()

    ####################################################################################################################

    t0 = time()

    print('Entrenamos el Pipeline con los datos de las opiniones de las películas...', end=' ')

    parameters = {
        'tfidf__ngram_range': [(1, 3)],
        'tfidf__smooth_idf': [True, False],
        'tfidf__use_idf': [True, False],
        'tfidf__sublinear_tf': [True, False],
        'tfidf__binary': [True, False],
        'multinomial__alpha': [0.1, 0.25, 0.5, 0.75, 1.0]
    }

    # Creamos el GridSearch para el Pipeline seleccionado
    gs_pipeline = GridSearchCV(pipeline, parameters, n_jobs=n_jobs)

    # Lo entrenamos con los vectores
    gs_pipeline.fit(x_train, y_train)

    print('terminado en {0:.2f}s'.format(time() - t0))
    print()

    ####################################################################################################################

    t0 = time()

    print('Clasificamos y mostramos el rendimiento del pipeline')

    # Clasificamos los vectores del conjunto de prueba
    prediction = gs_pipeline.predict(x_sample)

    # Clasificamos los comentarios de ejemplo introducidos
    for i, predict in enumerate(prediction):
        print('El comentario:')
        print('\t {0}'.format(x_sample[i]))
        print('\t Es {0}'.format(target_names[predict]))
        print()

    # Mostramos los datos del clasificador
    print('Los datos del clasificador son:')

    # Este mejor rendimiento se obtiene por medio de una validación cruzada, por tanto, no es necesario que la hagamos
    # explícitamente
    print('\t El mejor rendimiento obtenido con el conjunto entrenado es del: \n\t\t {0:.2f}%'.format(
        gs_pipeline.best_score_ * 100
    ))
    print('\t Los mejores parámetros seleccionados son: \n\t\t {0}'.format(
        gs_pipeline.best_params_
    ))

    print('Terminado en {0:.2f}s'.format(time() - t0))
    print()


if __name__ == '__main__':

    print('Iniciando la ejecución del programa principal...', end='\n\n')

    t0 = time()

    samples = [
        'Iron man movie it\'s fantastic.',
        'The movie is a little bored.',
        'It\'s a incredible film, I love it',
        'The movie is not bad for the low budget it has.',
        'In the beginning the movie was bored, but when Brad Pit kill zombies, it was amazing.'
    ]

    task_3(reviews_samples=samples, n_jobs=-1, training_taken=300)

    print('La ejecución total del programa ha sido de {0:.2f}s'.format(time() - t0))
