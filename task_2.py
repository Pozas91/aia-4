from utils import text


# =============================================================================
# CLASIFICADOR Naive Bayes Multinomial
# =============================================================================

class NaiveBayesMultinomial():
    
    def __init__(self):
        self.categorias = ['politica', 'deporte', 'sociedad']
        self.entrenamiento = [
            ['politica', 'Rajoy se encuentra en su peor momento.'],
            ['politica', 'Fraude hacienda'],
            ['deporte', 'Real Madrid campeón de la champions'],
            ['sociedad', 'Risto saca nuevo libro'],
            ['deporte', 'Cristiano Ronaldo renueva por 115 millones'],
            ['deporte', 'Rajoy se encuentra en su peor momento.'],
            ['sociedad', 'Rajoy se encuentra en su peor momento.']
        ]
    
    # P(c) se estima como Nc/N , donde Nc es el numero de documentos de la categoria c 
    # y N el numero total de documentos en el conjunto de entrenamiento.
    def probabilidad_documentos_por_categoria(self, categoria):
        contador = 0
        for x in self.entrenamiento:
            if x[0] == categoria:
                contador += 1
                
        return contador / len(self.entrenamiento)
    
    # P(t|c) es la proporcion de ocurrencias de t en todos los documentos de la categoria c 
    # (respecto de todas las ocurrencias de todos los terminos del vocabulario)
    def proporcion_ocurrencias_texto(self, texto, categoria):
        pass
        
    
    # Imprime las categorías y el conjunto de entrenamiento
    def imprime(self):
        print('Categorías: {0}'.format(self.categorias))
        print('Conjunto de entrenamiento: {0}'.format(self.entrenamiento))