import joblib
import cv2
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam


import capturaMao


def preprocess_image(image_path):
   input_img = cv2.imread(image_path)
   input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
   input_img_resize = cv2.resize(input_img, (28, 28),
                                 interpolation=cv2.INTER_AREA)  # redimensionando para (28, 28) usando cv2.INTER_AREA

   img_data = np.array(input_img_resize)
   img_data = np.expand_dims(img_data, axis=-1)  # adicionando dimensão de canal
   img_data = np.expand_dims(img_data, axis=0)  # adicionando dimensão de lote
   img_data = img_data.astype('float32')
   img_data /= 255

   return img_data


def plot_image(img_data):
   img = img_data.reshape(28, 28) # redimensionando para (28, 28)
   plt.imshow(img, cmap='gray')
   plt.axis('off')
   plt.get_current_fig_manager().window.state('zoomed')
   plt.show()

def predict_image(model, image_path):
  # Pré-processar a imagem
  img = preprocess_image(image_path)

  # Fazer a previsão
  prediction = model.predict(img)

  # Pegar o índice com a maior probabilidade
  predicted_digit = np.argmax(prediction)

  return predicted_digit

# Carregar os dados
(train_img, train_lab), (test_img, test_lab) = mnist.load_data()

# Redimensionar imagens e normalizar para valores entre 0 e 1
train_img = train_img.reshape((-1,28,28,1)) / 255.0
test_img = test_img.reshape((-1,28,28,1)) / 255.0

# Codificar labels para one-hot
train_lab = to_categorical(train_lab)
test_lab = to_categorical(test_lab)

# Definir o modelo
model = Sequential([
   Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1), name='conv2d_1'),
   MaxPooling2D((2, 2)),
   Conv2D(16, (5, 5), activation='relu', name='conv2d_2'),
   MaxPooling2D((2, 2)),
   Flatten(),
   Dense(120, activation='relu'),
   Dense(84, activation='relu'),
   Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo e armazenar o histórico
history = model.fit(train_img, train_lab, epochs=10, batch_size=128, validation_data=(test_img, test_lab))
img = preprocess_image("desenho.png")

plot_image(img)

# Visualização das saídas de camadas intermediárias
intermediate_layer_model_1 = Model(inputs=model.input,
                                   outputs=model.get_layer('conv2d_1').output)
intermediate_output_1 = intermediate_layer_model_1.predict(img)

intermediate_layer_model_2 = Model(inputs=model.input,
                                   outputs=model.get_layer('conv2d_2').output)
intermediate_output_2 = intermediate_layer_model_2.predict(img)


for i in range(intermediate_output_1.shape[3]):
    plt.subplot(6, 6, i+1)
    plt.imshow(intermediate_output_1[0, :, :, i], cmap='gray')
    plt.axis('off')
plt.get_current_fig_manager().window.state('zoomed')
plt.show()


for i in range(intermediate_output_2.shape[3]):
    plt.subplot(6, 6, i+1)
    plt.imshow(intermediate_output_2[0, :, :, i], cmap='gray')
    plt.axis('off')
plt.get_current_fig_manager().window.state('zoomed')
plt.show()

# Visualização dos filtros da camada convolucional
filtros, vieses = model.get_layer('conv2d_1').get_weights()

# Normaliza os valores do filtro para 0-1 para que possamos visualizá-los
f_min, f_max = filtros.min(), filtros.max()
filtros = (filtros - f_min) / (f_max - f_min)

n_filtros, ix = 6, 1
for i in range(n_filtros):
    # Pegando o filtro
    f = filtros[:, :, :, i]

    # Plotando cada canal separadamente
    for j in range(1):
        # Especificando subplot e desligando eixo
        ax = plt.subplot(n_filtros, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])

        # Plotando canal do filtro em escala de cinza
        plt.imshow(f[:, :, j], cmap='gray')
        ix += 1

# Mostrando a figura
plt.get_current_fig_manager().window.state('zoomed')
plt.show()


# Plotando perdas de treino e validação

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perda no Treino')
plt.plot(history.history['val_loss'], label='Perda na Validação')
plt.title('Perdas')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

# Plotando acurácias de treino e validação
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Acurácia no Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia na Validação')
plt.title('Acurácias')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.tight_layout()
plt.get_current_fig_manager().window.state('zoomed')
plt.show()

# Convertendo rótulos one-hot para formato de número inteiro
test_lab = np.argmax(test_lab, axis=1)

# Fazendo predições no conjunto de teste
test_predictions = model.predict(test_img)
test_predictions = np.argmax(test_predictions, axis=1)

# Calculando a acurácia
accuracy = accuracy_score(test_lab, test_predictions)
print(f'Acurácia no Teste: {accuracy*100:.2f}%')

# Calculando a matriz de confusão
cm = confusion_matrix(test_lab, test_predictions)

# Plotando a matriz de confusão
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.get_current_fig_manager().window.state('zoomed')
plt.show()


#Armazena a rede treinada.
joblib.dump(model, 'rede.joblib')

print(predict_image(model, 'desenho.png'))

