from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import pandas as pd
import os
from time import time

########################################################################################################################

tt = time()
print()

########################################################################################################################

t0 = time()

print('Cargando y pasando datos desde el csv...', end=' ')

movie_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/movie_data.csv'))

x = list()
y = list()
target_names = ['Negativo', 'Positivo']

for values in movie_data.values:
    review, sentiment = values
    x.append(review)
    y.append(sentiment)

print('terminado en {0:.2f}s'.format(time() - t0))
print()

########################################################################################################################

t0 = time()

print('Sacamos por validación cruzada el conjunto de entrenamiento y el de pruebas...', end=' ')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
x_sample = ['Iron man movie it\'s fantastic.', 'The movie is a little bored.']

print('terminado en {0:.2f}s'.format(time() - t0))
print()

########################################################################################################################

t0 = time()

print('Creamos el vectorizador con las opiniones de las películas...', end=' ')

vectorizer_parameters = {
    'ngram_range': [(1, 3)],
    'stop_words': ['english'],
    'smooth_idf': [True, False],
    'use_idf': [True, False],
    'sublinear_tf': [True, False],
    'binary': [True, False]
}

vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')

x_vectorizer_train = vectorizer.fit_transform(x_train)
x_vectorizer_test = vectorizer.transform(x_test)
x_vectorizer_sample = vectorizer.transform(x_sample)

print('terminado en {0:.2f}s'.format(time() - t0))
print()

########################################################################################################################

t0 = time()

print('Entrenamos las opiniones con el Multinomial de Naive Bayes...', end=' ')

mnb_parameters = {
    'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]
}

# Creamos el GridSearch para el MultinomialNB
gs_mnb = GridSearchCV(MultinomialNB(), mnb_parameters)

# Lo entrenamos con los vectores
gs_mnb.fit(x_vectorizer_train, y_train)

print('terminado en {0:.2f}s'.format(time() - t0))
print()

########################################################################################################################

t0 = time()

print('Clasificamos y mostramos el rendimiento del Multinomial de Naive Bayes')

# Clasificamos los vectores del conjunto de prueba
prediction = gs_mnb.predict(x_vectorizer_sample)

# Clasificamos los comentarios de ejemplo introducidos
for i, predict in enumerate(prediction):
    print('El comentario:')
    print('\t {0}'.format(x_sample[i]))
    print('\t Es {0}'.format(target_names[predict]))
    print()

# Comprobamos el rendimiento obtenido
score_gs_mnb = gs_mnb.score(x_vectorizer_test, y_test)

print('Los datos del clasificador son:')
print('\t El rendimiento con el conjunto entrenado es del {0:.2f}%'.format(score_gs_mnb * 100))
print('\t Los mejores parámetros seleccionados son: {0}'.format(gs_mnb.best_params_))

print('Terminado en {0:.2f}s'.format(time() - t0))
print()

########################################################################################################################

print('La ejecución total del programa ha sido de {0:.2f}s'.format(time() - tt))
