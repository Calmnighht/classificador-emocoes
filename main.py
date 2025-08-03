# Base de dados simples de frases com emoções
frases = [
    ("Estou tão feliz hoje!", "alegria"),
    ("Que dia Maravilhoso!", "alegria"),
    ("Isso me deixa com raiva.", "raiva"),
    ("O Transito me estressa muito.", "raiva"),
    ("Estou tão triste com essa notícia.", "tristeza"),
    ("Sinto uma grande tristeza por isso.", "tristeza"),
    ("Sinto falta dos velhos tempos.", "tristeza"),
    ("Estou tão animado para o futuro!", "felicidade"),
    ("Hoje acordei animado!", "alegria"),
    ("Estou tão frustrado com esse projeto.", "frustração"),
    ("Não consigo acreditar que isso aconteceu.", "surpresa"),
     
]

# Separar as frases e rotulos (emoções)
x = [x[0] for x in frases]
y = [x[1] for x in frases]

from sklearn.feature_extraction.text import CountVectorizer
# Criar o vetor de palavras
vectorizer = CountVectorizer()
X_vetorizado = vectorizer.fit_transform(x)

from sklearn.naive_bayes import MultinomialNB

# Criar o modelo e treinar
modelo= MultinomialNB()
modelo.fit(X_vetorizado, y)

# Testar com uma nova frase
frase_teste = ["Estou me sentindo muito bem hoje"]
frase_vetorizada = vectorizer.transform(frase_teste)

import numpy as np

# Fazer a predição
probas = modelo.predict_proba(frase_vetorizada)[0]

#mostrar porcentagem de confiança para cada emoção
for emoção, prob in zip(modelo.classes_, probas):
    print(f"{emoção}: {prob * 100:.2f}%")

# mostrar a emoção com maior probabilidade
indice_mais_confiante = np.argmax(probas)
print(f"\nEmoção prevista: {modelo.classes_[indice_mais_confiante]}")

import matplotlib.pyplot as plt

#criar gráfico de barras
plt.bar(modelo.classes_, probas, color='skyblue')
plt.title('Confiança da IA para cada emoção')
plt.xlabel('Emoções')
plt.ylabel('Confiança (%)')
plt.ylim(0, 1)
plt.grid(True)
plt.show()