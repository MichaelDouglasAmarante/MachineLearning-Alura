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

# 1 => porco, 0 => cachorro
dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
classes = [1,1,1,0,0,0] 


# instanciando um modelo/estimador
model = LinearSVC()

# aprendizado supervisionado
model.fit(dados, classes)


# testando com animais misteriosos
misterio1 = [1,1,1] # cachorro
misterio2 = [1,1,0] # porco
misterio3 = [0,1,1] # porco

testes = [misterio1, misterio2, misterio3]
previsoes = model.predict(testes) 

testes_classes = [0, 1, 1]

corretos = (previsoes == testes_classes).sum() # acertou 2 de 3
total = len(testes)
taxa_de_acerto = corretos / total
print("Taxa de acerto: ", taxa_de_acerto * 100)

from sklearn.metrics import accuracy_score
taxa_de_acerto = accuracy_score(testes_classes, previsoes)

