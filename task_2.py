# =============================================================================
# CLASIFICADOR Naive Bayes Multinomial
# =============================================================================


class NaiveBayesMultinomial:

    def __init__(self):
        self.k = 0
        self.categories = ['politica', 'deporte', 'sociedad']
        self.x_train = [
            ['politica', 'Jordi Sànchez anuncia ante el Supremo su disposición a dejar el escaño'],
            ['politica', 'Serret confirma que Jordi Turull será candidato y luego rectifica'],
            ['politica', 'El fiscal pide invalidar los pasaportes de Puigdemont y del resto de huidos'],
            ['politica', 'El TC vuelve a rechazar la petición del PSC para desbloquear la investidura'],
            ['politica', 'El plan a la desesperada de la ANC para hacer frente al 155'],
            ['politica',
             'Romeva autorizó que la misión de observadores viajaran en primera clase y “alto standing” por imagen'],
            ['politica',
             'Sánchez afea a Rajoy como “intermediario de la nada” ante los “dramas” y “tragedias” sociales'],
            ['politica', 'Cs abandona la Comisión Territorial del Congreso impulsada por el PSOE'],
            ['politica', 'PP y C’s avalan en el Congreso el veto del Gobierno a la ley de la PAH'],
            ['politica', 'El histórico etarra ‘Kubati’, entre los detenidos por homenajear a una presa de la banda'],
            ['politica',
             'Montoro defiende pedir informes de empresas afines al procés ante las acusaciones de “macartismo”'],
            ['politica',
             'Un eurodiputado de ERC denuncia la “violencia” del 1-O en un pleno del Consejo de Derechos Humanos de la ONU'],
            ['politica',
             'La Audiencia Nacional da tres días a Valtonyc para pedir la suspensión de condena antes de ser encarcelado'],
            ['politica', 'Cifuentes niega la mayor para sacudirse la corrupción del PP de Madrid'],
            ['politica', 'Carles Agustí apuesta por una lista conjunta para Barcelona con ERC, Graupera y vecinos'],
            ['politica', 'Últimas noticias de Catalunya y de Jordi Sànchez, en directo'],
            ['politica', 'Los independentistas trabajan para una investidura en Semana Santa'],
            ['politica', 'El PDECat reforzará su estructura con una portavoz'],
            ['politica', 'Feijóo en el Guadarrama'],
            ['politica', 'La ley de amnistía de los crímenes franquistas distancia de nuevo al PSOE de Podemos'],
            ['politica', 'El PP vasco muestra fotos de víctimas de ETA en el recibimiento a un preso'],
            ['politica',
             'Puigdemont y Gabriel, juntos en un debate en la ONU contra “la regresión de los derechos humanos en España”'],
            ['politica', 'Colau mantiene la votación sobre el tranvía y retrasa la de la ‘multiconsulta’'],
            ['politica', 'El Parlament tramita la reforma de la ley para permitir una investidura a distancia'],
            ['politica', 'Torrent responde a Arrimadas: “Mi deber es defender los derechos” de Puigdemont y Sànchez'],
            ['politica', 'Carmena califica la actuación municipal de correcta y cierra filas con la Policía'],
            ['politica', 'El PP descarta que Rajoy convoque elecciones en Catalunya'],
            ['politica', 'Puigdemont visitará Finlandia esta semana'],
            ['politica', 'Un sindicato policial de Madrid se querella contra una edil de Carmena por incitar al odio'],
            ['politica', 'ERC señala la comparecencia de Sànchez ante el TS como “día clave” para la investidura'],
            ['politica', 'Díez Usabiaga (Sortu): “ETA no teme reconocer lo positivo y negativo de su historia”'],
            ['politica', 'La concejala de la CUP de Reus investigada por odio no acude a declarar y será detenida'],
            ['politica', 'Popcoin: la nueva forma de invertir abierta a todo el mundo'],
            ['politica', 'Catalunya, en un vacío legal tras la resolución del TC y el aplazamiento de la investidura'],
            ['politica', 'Ciudadanos atrapa al PP mientras el PSOE se estanca y Podemos se hunde'],
            ['politica',
             'Los ‘casals’ catalanes avisan de que tendrán dificultades si la escasez de subvenciones por el 155 se alarga'],
            ['politica', 'Iceta urge a formar un Govern “dentro de la legalidad”'],
            ['politica', 'El PP desbloquea la reforma de la Ley de Secretos Oficiales como pedía el PNV'],
            ['politica',
             'El PNV y los padres, profesores y alumnos de la escuela pública se van también del pacto educativo'],
            ['politica', 'Ciudadanos abre la vía para que los funcionarios estatales y autonómicos cobren lo mismo'],
            ['politica', 'Encuentran el cuerpo sin vida del guardia civil desaparecido en Guillena'],
            ['politica', 'Cifuentes endosa a Génova la falta de actuación contra la corrupción en Madrid'],
            ['politica', 'Pleno del Congreso de los Diputados, en directo'],
            ['politica', 'Condenado a 21 meses de cárcel por controlar, aislar y humillar a su mujer durante 49 años'],
            ['politica', 'Jordi Sànchez anuncia que está dispuesto a renunciar a la política a cambio de su libertad'],
            ['politica', 'Otros 76.000 españoles se marcharon al extranjero en 2017 y ya son más de 2,5 millones'],
            ['politica',
             'Podemos pide que las niñas puedan elegir en Aragón entre pantalón y falda para el uniforme del colegio'],
            ['politica', 'Detenido un hombre en Lugo acusado de matar a su padre a golpes'],
            ['politica', 'Ciudadanos quiere que las comunidades de vecinos puedan ordenar desalojos de pisos'],
            ['politica', 'Este año San José es sólo festivo en la Comunidad Valenciana y Murcia'],
            ['deporte', 'Las 15 jugadas más polémicas del robo a España en Bélgica'],
            ['deporte', 'Lucas Hernández, Rafinha... y Messi: ocho futbolistas que pudieron jugar con España'],
            ['deporte', '¿Cuánto cuesta el once revelación de la Liga? Valores, cláusulas y chollos'],
            ['deporte', 'Irónico pero real: la figura del padre-entrenador que avergüenza a los niños'],
            ['deporte', 'Querido Barça, ¿90 euros la entrada para el partido de la Roma en Champions?'],
            ['deporte', 'Feddal comienza la rehabilitación en la Ciudad Deportiva'],
            ['deporte', 'Paco Herrera: "De los 3 de abajo, la UD es el equipo con más opciones"'],
            ['deporte', 'Imanol ya se hace oír en Zubieta'],
            ['deporte', 'Zidane toma nota del triplete del Girona a balón parado pensando en la Juventus'],
            ['deporte', 'Ziganda da tregua a Beñat y Aduriz'],
            ['deporte', 'Valverde recupera a Denis Suárez y Nélson Semedo'],
            ['deporte', 'El mercado de enero, un fiasco'],
            ['deporte', 'Sergi, canterano del Atlético, sufre un esguince en su tobillo izquierdo'],
            ['deporte', 'Jarvis Varnado, el taponador supremo, abandona el Zaragoza'],
            ['deporte', 'Livio Jean-Charles ya está en Málaga'],
            ['deporte', 'Stephen Curry apunta a los Hawks para volver a las canchas'],
            ['deporte', 'Cuatro partidos a los que apostar este martes en la Euroliga'],
            ['deporte', 'Draymond Green recibe de su propia medicina: K.O. tras un golpe bajo'],
            ['deporte', 'La enigmática reportera que se ha vuelto viral en la Locura de Marzo de la NCAA'],
            ['deporte', 'LeBron James, un MVP contra la lógica y el tiempo'],
            ['deporte', 'Resúmenes y resultados de la jornada NBA: todos los partidos'],
            ['deporte', 'La peor versión de Lonzo Ball sucumbe ante el LeBron de Brooklyn'],
            ['deporte', 'Stephen Curry apunta a los Hawks para volver a las canchas'],
            ['deporte', 'Los Grizzlies confían en que un "mal año" no les impida retener a Marc Gasol'],
            ['deporte', 'La fiesta más loca de Embiid: 12.000 dólares en alcohol con uno de sus jefes'],
            ['deporte',
             'El impresionante discurso de un entrenador de baloncesto sobre los padres en los deportes base'],
            ['deporte', 'El homenaje capilar más friqui de la NBA'],
            ['deporte', 'Messi y Jordan se encarnan en el hijo de dos leyendas de la WNBA'],
            ['deporte', 'Steve Francis, ex estrella de la NBA, arrestado por estar borracho en público'],
            ['deporte', 'Ricky Rubio: "Estoy contento con el equipo y lo mejor es que vamos a más"'],
            ['deporte', 'Pedrosa, otra vez con problemas de neumáticos'],
            ['deporte', 'Crivillé: "Dovi acertó con la mejor estrategia"'],
            ['deporte', 'Martín se hace mayor'],
            ['deporte', 'Viñales, al fin, encuentra el camino'],
            ['deporte', 'Las mejores imágenes del GP de Qatar'],
            ['deporte', 'Rossi: "Es un poco pronto para decir si lucharé por el título"'],
            ['deporte', 'Dovizioso: "He vuelto a batir a Marc, estoy muy contento"'],
            ['deporte', 'Marc Márquez: "Lo intenté con todo, algún día lo conseguiré"'],
            ['deporte', 'Dovizioso gana un pulso espectacular a Márquez'],
            ['deporte', 'Cuatro partidos a los que apostar este martes en la Euroliga'],
            ['deporte', 'Livio Jean-Charles ya está en Málaga'],
            ['deporte', 'El Real Madrid y Rudy Fernández comienzan a negociar la renovación'],
            ['deporte', 'Slaughter celebra su cumpleaños como el mayor aficionado del Madrid'],
            ['deporte', 'Doncic podría volver ya en Valencia y Llull, en semanas'],
            ['deporte', 'Heurtel-Claver la conexión que pone el espectáculo en el Barça de Pesic'],
            ['deporte', 'El PAO de Xavi Pascual condena virtualmente al Unicaja'],
            ['deporte', 'Prodigio de Melli que anota desde su casa'],
            ['deporte', 'Fenerbahce vs CSKA: De Colo ejecuta al campeón y el CSKA certifica su liderato'],
            ['deporte', 'El Barcelona da un recital en Atenas para despedazar al Olympiacos'],
            ['deporte', 'El Valencia se despide de la Euroliga en la sede de la Final Four'],
            ['sociedad',
             'Juana Rivas viaja a Italia para reencontrarse con sus hijos y acudir a la vista sobre su custodia'],
            ['sociedad', 'Un ex ministro de Mitterrand acosa sexualmente a la hija de un político'],
            ['sociedad', 'Las denuncias por violencia de género aumentaron un 18% de abril a junio'],
            ['sociedad', 'Fallece la persona atrapada en el derrumbamiento de Arazuri'],
            ['sociedad', 'La última oportunidad de Ibar'],
            ['sociedad', 'Fallece una anciana de 88 años tras un incendio en una residencia geriátrica en Cartagena'],
            ['sociedad', 'Anna y la dignidad'],
            ['sociedad',
             'Prisión provisional sin fianza para el hombre que presuntamente acabó con la vida de su mujer en Azuqueca'],
            ['sociedad', 'Profesores de puertas abiertas'],
            ['sociedad', 'Un médico británico golpea a un tiburón en Australia para liberarse de su ataque'],
            ['sociedad', 'Una madre y su hija, heridas en un accidente durante una prueba de rally en Campanet'],
            ['sociedad', 'Investigan al padre de un niño de 12 años por dejarle conducir su taxi'],
            ['sociedad', 'Morate estuvo ilocalizable seis horas después de los asesinatos de Laura y Marina'],
            ['sociedad',
             'La ex novia de Morate fue asfixiada con una brida que redujo el diámetro de su cuello de 23 a 8 centímetros'],
            ['sociedad', 'Cuatro detenidos por una agresión sexual múltiple a una turista en Canarias'],
            ['sociedad', 'El Gobierno augura un Pacto de Estado de Educación de mínimos'],
            ['sociedad', 'Refugiados por su orientación sexual: los gays marroquíes del CETI de Ceuta'],
            ['sociedad', 'Jucio contra Morate: "Marina se sentía perseguida, tenía miedo y sufría violencia física"'],
            ['sociedad',
             'En libertad la mujer de 74 años detenida por un incendio en la localidad pontevedresa de Mos'],
            ['sociedad', '¿Hay que extender la enseñanza obligatoria en España desde los 16 hasta los 18 años?'],
            ['sociedad',
             'Marruecos repele un intento de entrada en Ceuta por la frontera de unos 50 migrantes subsaharianos'],
            ['sociedad', 'Investigan a una segunda sospechosa por la oleada de incendios en Galicia'],
            ['sociedad', 'Un diplomático saudí pierde la inmunidad por maltratos al servicio doméstico'],
            ['sociedad', 'El gesto solidario de una catalana con los afectados por los incendios en Galicia'],
            ['sociedad', 'Mitología, leyendas y alguna verdad sobre las causas de los incendios en Galicia'],
            ['sociedad', 'La sequía de la década, no del siglo'],
            ['sociedad', 'España y Portugal se coordinarán para activar el fondo de la UE ante incendios'],
            ['sociedad', 'El director de seguridad de Renfe, investigado por el accidente del Alvia'],
            ['sociedad',
             'El jurado declara a Sergio Morate culpable de los asesinatos de Marina Okarynska y Laura del Hoyo'],
            ['sociedad', 'Anna Veiga: "Una pareja nos pidió que clonáramos a su hijo muerto"'],
            ['sociedad', 'Innovación cultural en el camino correcto y de forma disruptiva para derribar prejuicios'],
            ['sociedad', 'La primavera será más cálida en el sur y normal en lluvias'],
            ['sociedad',
             'El uso del móvil dispara el síndrome del cuello roto: los músculos soportan el equivalente a 25 kilos'],
            ['sociedad',
             'Un hipopótamo sorprende a los vecinos de un pueblo de Badajoz tras escaparse de un circo ambulante'],
            ['sociedad', 'La Semana Santa arrancará con frío y lluvia en todo el país'],
            ['sociedad', 'La Iglesia recibe más fondos que nunca del IRPF pero pierde 500.000 contribuyentes'],
            ['sociedad', 'El mejor consejo contra la astenia primaveral: alimentación sana y descanso'],
            ['sociedad', 'Un ensayo clínico muestra que la píldora anticonceptiva masculina es segura y eficaz'],
            ['sociedad', 'Un ensayo clínico muestra que la píldora anticonceptiva masculina es segura y eficaz'],
            ['sociedad', 'El TC se inclina por avalar las subvenciones públicas a centros que segregan por sexos'],
            ['sociedad', 'Cinco medidas de la Lomce que ya no se aplican'],
            ['sociedad', 'Podemos pide que no se obligue a llevar uniformes diferenciados por sexo'],
            ['sociedad', 'El PP quiere que el Congreso elabore un informe sobre Movilidad Sostenible'],
            ['sociedad', 'Las aspiraciones de España en el sector de los drones: un negocio de 1.200 millones'],
            ['sociedad', 'El emotivo karaoke de 50 niños con síndrome de Down cantando junto a sus madres'],
            ['sociedad', 'El islote privado ibicenco de sEspalmador, vendido a un particular por 18 millones de euros'],
            ['sociedad', 'Alerta por la nueva estafa sobre Mercadona de la que avisa la Policía'],
            ['sociedad', 'Aprender a utilizar un desfibrilador'],
            ['sociedad', '20.000 sonidos bajo el mar'],
            ['sociedad', 'Inmensidad de datos']
        ]
        self.log_p = list()
        self.log_ptc = list()

    def count_total_words(self) -> int:
        """
        Obtiene el total de palabras en el conjunto de entrenamiento sin filtrar (|V|)
        :return:
        """
        return sum([len(sentence.split()) for _, sentence in self.x_train])

    def total_words(self):

        """
        Devuelve un diccionario con todas las palabras que se encuentran en el conjunto de entrenamiento
        :return:
        """
        total_words = dict()

        for _, sentence in self.x_train:

            words = sentence.split()

            for word in words:
                if word in total_words:
                    total_words[word] += 1
                else:
                    total_words[word] = 1

        return total_words

    def documents_per_category_probability(self, category):

        """
        P(c) se estima como Nc/N, donde Nc es el número de documentos de la categoría c y N el número total de
        documentos en el conjunto de entrenamiento.
        :param category:
        :return:
        """

        count = sum([1 for x, _ in self.x_train if x == category])

        return count / len(self.x_train)

    def text_occurrences_proportion(self, word, category):
        """
        P(t|c) es la proporción de ocurrencias de t en todos los documentos de la categoría c
        (respecto de todas las ocurrencias de todos los términos del vocabulario
        :param word:
        :param category:
        :return:
        """
        filtered_set = [x for x in self.x_train if x[0] == category]

        counts = dict()
        total_words = 0
        total_words_category = 0
        probability = 0

        for sentences in filtered_set:

            words = sentences[1].split()

            for x in words:
                if x in counts:
                    counts[x] += 1
                else:
                    counts[x] = 1

            # Aumento el contador de palabras iguales
            if counts.get(word) is not None and counts.get(word) != 0:
                total_words += counts.get(word)

            # Total de palabras en el conjunto de entrenamiento filtrado por la categoría
            total_words_category += len(sentences[1].split())

            if self.k != 0:
                probability = (total_words + self.k) / (total_words_category + self.k * self.count_total_words())
            else:
                probability = total_words / total_words_category

        return probability

    def calc_log_p(self):
        p_categories = list()

        for category in self.categories:
            p_categories.append(self.documents_per_category_probability(category))

        self.log_p = p_categories

    def calc_log_ptc(self):
        p_categories = list()

        for category in self.categories:
            total_words = self.total_words()
            word_by_category = dict()

            for word in total_words:
                word_by_category[word] = self.text_occurrences_proportion(word, category)

            p_categories.append(word_by_category)

        self.log_ptc = p_categories

    def fit(self):
        """
        Método para entrenar el clasificador
        :return:
        """
        self.calc_log_p()
        self.calc_log_ptc()

    def predict(self, headline):

        """
        Predice el titular pasado
        :param headline:
        :return:
        """

        best_score = 0
        best_category = ''

        for category in self.categories:

            log_p, log_ptc = 0, 0

            if category == 'politica':
                log_p = self.log_p[0]
            elif category == 'deporte':
                log_p = self.log_p[1]
            else:
                log_p = self.log_p[2]

            for word in headline.split():

                if category == 'politica':
                    if self.log_ptc[0].get(word) is not None and self.log_ptc[0].get(word) != 0:
                        log_ptc += self.log_ptc[0].get(word)
                elif category == 'deporte':
                    if self.log_ptc[1].get(word) is not None and self.log_ptc[1].get(word) != 0:
                        log_ptc += self.log_ptc[1].get(word)
                else:
                    if self.log_ptc[2].get(word) is not None and self.log_ptc[2].get(word) != 0:
                        log_ptc += self.log_ptc[2].get(word)

            total_sum_by_category = log_p + log_ptc

            if total_sum_by_category > best_score:
                best_score = total_sum_by_category
                best_category = category

        return best_category

    def change_k_value(self, k):
        """
        Cambia el valor de k por defecto
        :param k:
        :return:
        """
        self.k = k

    def score(self, x_test):

        """
        Imprime el porcentaje de aciertos del clasificador
        :param x_test:
        :return:
        """

        successfully_answers = 0
        correctly_classified = list()
        incorrectly_classified = list()

        for headline in x_test:
            category = self.predict(headline[1])

            if category == headline[0]:
                successfully_answers += 1
                correctly_classified.append([category, headline[1]])
            else:
                incorrectly_classified.append([category, headline[1]])

        score = successfully_answers / len(x_test)

        print('Titulares clasificados correctamente: ')

        for headline in correctly_classified:
            print('\tClasificado: {0} --> Titular: {1}'.format(
                headline[0], headline[1]
            ))

        print('\n\nTitulares clasificados incorrectamente: '.format(
            incorrectly_classified
        ))

        for headline in incorrectly_classified:
            print('\tClasificado: {0} --> Titular: {1}'.format(
                headline[0], headline[1]
            ))

        print('\n\nPorcentaje de aciertos: {0}% ({1}/{2})'.format(
            score, successfully_answers, len(x_test)
        ))

    def print(self):
        """
        Imprime las categorías y el conjunto de entrenamiento
        :return:
        """
        print('Categorías: {0}'.format(self.categories))
        print('Conjunto de entrenamiento: {0}'.format(self.x_train))
        print('Tamaño conjunto de entrenamiento: {0}'.format(len(self.x_train)))


# =============================================================================
# PRUEBAS TASK 2
# =============================================================================
NaiveBayes = NaiveBayesMultinomial()

# =============================================================================
# ENTRENAMIENTO
# =============================================================================
NaiveBayes.fit()

# =============================================================================
# PREDECIMOS
# =============================================================================
x_train = [
    ['deporte', '¿Frenar a Douglas Costa o pensar en el Mundial? Casemiro no duda: "Le parto en dos"'],
    ['politica', 'Esquerra Unida, todavía sin coordinadora: la militancia se divide entre Rosa Pérez y Rosa Albert'],
    ['deporte', 'Florentino: "Si tienes un amigo en Hong Kong, le preguntas por Hong Kong"'],
    ['politica',
     'Rivera sobre la manifestación del 8M: "Todo el mundo estuvo junto y unido para defender la libertad y la igualdad de las mujeres"'],
    ['sociedad', 'El Gobierno confirma para abril un trasvase del Tajo al Segura'],
    ['sociedad', 'Un mensaje de Whatsapp enviado por error acaba en boda'],
    ['sociedad',
     'El catalán se usará de modo «prioritario» en «todos los ámbitos» de la Universidad de las Islas Baleares'],
    ['sociedad', 'Ciudad del Cabo, la primera gran urbe que se enfrenta a vivir sin agua'],
    ['deporte', 'Javi Fernández abandera a España en el adiós a los «Juegos de la Paz»'],
    ['politica',
     'Rivera sobre la manifestación del 8M: "Todo el mundo estuvo junto y unido para defender la libertad y la igualdad de las mujeres"'],
]

# =============================================================================
# MOSTRAMOS EL RENDIMIENTO DEL CLASIFICADOR
# =============================================================================
NaiveBayes.score(x_train)
