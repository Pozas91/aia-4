import task_2


# =============================================================================
# PRUEBAS TASK 2
# =============================================================================
NaiveBayes = NaiveBayesMultinomial()


# =============================================================================
# ENTRENAMIENTO
# =============================================================================
NaiveBayes.logPTC()
NaiveBayes.logP()

# =============================================================================
# PREDECIMOS
# =============================================================================

NaiveBayes.predict('Rajoy gana las elecciones')

NaiveBayes.predict('Baloncesto deporte estrella')

NaiveBayes.predict('McLaren estaría por delante de Renault')

NaiveBayes.predict('Rivera sobre la manifestación del 8M: "Todo el mundo estuvo junto y unido para defender la libertad y la igualdad de las mujeres"')