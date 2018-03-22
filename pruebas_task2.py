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

NaiveBayes.predict('¿Frenar a Douglas Costa o pensar en el Mundial? Casemiro no duda: "Le parto en dos"')

NaiveBayes.predict('Esquerra Unida, todavía sin coordinadora: la militancia se divide entre Rosa Pérez y Rosa Albert')

NaiveBayes.predict('Florentino: "Si tienes un amigo en Hong Kong, le preguntas por Hong Kong"')

NaiveBayes.predict('Rivera sobre la manifestación del 8M: "Todo el mundo estuvo junto y unido para defender la libertad y la igualdad de las mujeres"')