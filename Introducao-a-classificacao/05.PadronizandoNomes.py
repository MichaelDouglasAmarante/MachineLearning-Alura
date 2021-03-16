from sklearn.svm import LinearSVC

# features : caracteristicas
# pelo longo?
# perna curta?
# faz auau?
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

# f(x) = y
treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1,1,1,0,0,0] # também chamadas de labels / etiquetas

# instanciando um modelo/estimador
model = LinearSVC()

# aprendizado supervisionado
model.fit(treino_x, treino_y)

# testando com animais misteriosos
misterio1 = [1,1,1] # cachorro
misterio2 = [1,1,0] # porco
misterio3 = [0,1,1] # porco

teste_x = [misterio1, misterio2, misterio3]
teste_y = [0, 1, 1]
previsoes = model.predict(teste_x) 


# taxa de acerto com accuracy_score
from sklearn.metrics import accuracy_score
taxa_de_acerto = accuracy_score(teste_y, previsoes)
print("Taxa de acerto: %.2f" % (taxa_de_acerto * 100), "%")

