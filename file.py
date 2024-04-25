# Importações necessárias para análise de áudio, aprendizado de máquina, e visualização
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics

# Carregar o modelo YAMNet do TensorFlow Hub
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)  # Carregar o modelo YAMNet

# Função para adicionar ruído ao áudio
def add_noise(data, noise_factor=0.02):
    noise = noise_factor * np.random.randn(len(data))  # Gerar ruído
    return data + noise  # Adicionar ruído ao áudio

# Função para alterar a velocidade do áudio
def change_speed(data, speed_factor=0.8):
    return librosa.effects.time_stretch(data, rate=speed_factor)  # Alterar a velocidade

# Função para mudar o pitch do áudio
def change_pitch(data, pitch_steps=4):
    return librosa.effects.pitch_shift(data, sr=16000, n_steps=pitch_steps)  # Alterar o pitch

# Função para adicionar eco ao áudio
def add_echo(data, delay=0.2, decay=0.5):
    sr = 16000  # Taxa de amostragem
    delay_samples = int(delay * sr)  # Calcular a quantidade de atraso
    echo = np.zeros(len(data) + delay_samples)  # Criar array para o eco
    echo[:len(data)] = data  # Adicionar o áudio original
    echo[delay_samples:] += data * decay  # Adicionar o eco
    return echo  # Retornar o áudio com eco

# Função para modulação de amplitude
def modulate_amplitude(data, modulation_frequency=1.0):
    sr = 16000
    time = np.linspace(0, len(data) / sr, len(data))  # Criar vetor de tempo
    modulation = 1.0 + 0.5 * np.sin(2 * np.pi * modulation_frequency * time)  # Gerar modulação
    return data * modulation  # Modular a amplitude

# Função para mudar o volume do áudio
def change_volume(data, volume_factor=0.8):
    return data * volume_factor  # Multiplicar pelo fator para alterar o volume

# Função para inverter o áudio
def reverse_audio(data):
    return data[::-1]  # Inverter a ordem dos dados para reverter o áudio

# Função para aplicar várias técnicas de aumento de dados ao áudio
def apply_augmentations(audio):
    if np.random.rand() < 0.5:
        volume_factor = np.random.uniform(0.7, 1.3)
        audio = change_volume(audio, volume_factor)  # Alterar volume aleatoriamente
    if np.random.rand() < 0.3:
        audio = add_echo(audio, delay=0.1, decay=0.6)  # Adicionar eco
    if np.random.rand() < 0.3:
        audio = modulate_amplitude(audio, modulation_frequency=0.5)  # Modulação
    if np.random.rand() < 0.3:
        audio = reverse_audio(audio)  # Reversão
    if np.random.rand() < 0.5:
        audio = add_noise(audio, noise_factor=0.03)  # Adicionar ruído
    if np.random.rand() < 0.5:
        audio = change_speed(audio, speed_factor=0.9)  # Alterar velocidade
    if np.random.rand() < 0.5:
        audio = change_pitch(audio, pitch_steps=2)  # Alterar o pitch
    
    return audio  # Retornar o áudio com os augmentations aplicados

# Função para extrair MFCC do áudio
def extract_mfcc(audio, sr=16000, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)  # Extrair MFCC
    mfcc_mean = np.mean(mfcc, axis=1)  # Calcular a média para uma saída estática
    return mfcc_mean  # Retornar a média dos MFCC

# Função para extrair embeddings do YAMNet
def extract_yamnet_embeddings(audio):
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)  # Converter para tensor
    waveform = tf.reshape(waveform, [-1])  # Reformatar para waveform
    _, embeddings, _ = yamnet_model(waveform)  # Obter embeddings do YAMNet
    yamnet_mean = tf.reduce_mean(embeddings, axis=0).numpy()  # Calcular a média dos embeddings
    return yamnet_mean  # Retornar os embeddings do YAMNet

# Função para combinar MFCC e YAMNet em um único vetor
def extract_combined_features(audio):
    augmented_audio = apply_augmentations(audio)  # Aplicar augmentations
    mfcc_mean = extract_mfcc(augmented_audio)  # Extrair MFCC
    yamnet_mean = extract_yamnet_embeddings(augmented_audio)  # Extrair YAMNet
    
    combined_features = np.concatenate([mfcc_mean, yamnet_mean])  # Combinar MFCC e YAMNet
    return combined_features  # Retornar o vetor combinado

# Configurar caminhos para treinamento
base_dir = 'archive'  # Diretório base
train_audio_dir = os.path.join(base_dir, 'Train_submission/Train_submission/')  # Caminho do áudio de treinamento
metadata_train_path = os.path.join(base_dir, 'Metadata_Train.csv')  # Caminho dos metadados

# Carregar metadados do conjunto de treinamento
metadata_train = pd.read_csv(metadata_train_path)  
features = []
labels = []

# Iterar pelos metadados para extrair características e rótulos
for index, row in metadata_train.iterrows():
    file_name = row['FileName']  # Nome do arquivo de áudio
    file_path = os.path.join(train_audio_dir, file_name)  # Caminho do arquivo

    try:
        print("Processando arquivo de treino:", file_name)  # Exibir o arquivo processado
        audio, sr = librosa.load(file_path, sr=16000, mono=True)  # Carregar áudio
        combined_features = extract_combined_features(audio)  # Extrair características combinadas
        features.append(combined_features)  # Adicionar ao conjunto de características
        labels.append(row['Class'])  # Adicionar rótulo ao conjunto de rótulos
    except Exception as e:
        print(f'Erro ao processar o arquivo {file_name}: {e}')  # Tratar exceções

# Função para corrigir rótulos incorretos
def correct_labels(label):
    corrections = {
        'Sound_Guiatr': 'Sound_Guitar'  # Correção do erro de digitação
    }
    return corrections.get(label, label)  # Corrigir o rótulo ou retornar o original

# Corrigir rótulos e codificar
corrected_labels = [correct_labels(label) for label in labels]
le = LabelEncoder()  # Inicializar LabelEncoder
labels_encoded = le.fit_transform(corrected_labels)  # Codificar rótulos corrigidos
labels_onehot = tf.keras.utils.to_categorical(labels_encoded)  # Codificação one-hot

# Divisão do conjunto para treino e validação
X_train, X_val, y_train, y_val = train_test_split(np.array(features), np.array(labels_onehot), test_size=0.2, random_state=42)

# Modelo para combinar MFCC e YAMNet
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1024 + 13,)),  # Camada de entrada para MFCC + YAMNet
    tf.keras.layers.Dense(256, activation='relu'),  # Camada densa com ReLU
    tf.keras.layers.Dropout(0.3),  # Dropout para regularização
    tf.keras.layers.Dense(128, activation='relu'),  
    tf.keras.layers.Dropout(0.2),  # Outro Dropout para regularização
    tf.keras.layers.Dense(64, activation='relu'),  
    tf.keras.layers.Dropout(0.2),  # Dropout adicional
    tf.keras.layers.Dense(len(np.unique(labels_encoded)), activation='softmax')  # Saída Softmax para classes
])

# Configurar otimizador e compilar o modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adam com taxa de aprendizado de 0.001
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callback para parada precoce durante o treinamento
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True  # Parar se não houver melhoria após 10 épocas
)

# Treinamento do modelo com dados de validação
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[early_stopping])

# Avaliação do modelo após treinamento
val_loss, val_accuracy = model.evaluate(X_val, y_val)  
print(f'Perda de Validação: {val_loss}')  # Exibir a perda no conjunto de validação
print(f'Precisão de Validação: {val_accuracy}')  # Exibir a precisão no conjunto de validação

# Configurar caminhos para teste
metadata_test_path = os.path.join(base_dir, 'Metadata_Test.csv')  # Caminho dos metadados de teste
test_audio_dir = os.path.join(base_dir, 'Test_submission/Test_submission/')  # Diretório do áudio de teste
metadata_test = pd.read_csv(metadata_test_path)  # Carregar metadados do conjunto de teste

test_features = []
true_labels = []

# Processar arquivos de teste para extrair características e rótulos
for index, row in metadata_test.iterrows():
    file_name = row['FileName']  # Nome do arquivo de teste
    file_path = os.path.join(test_audio_dir, file_name)  # Caminho do arquivo
    
    try:
        print("Processando arquivo de teste:", file_name)  # Exibir o arquivo sendo processado
        audio, sr = librosa.load(file_path, sr=16000, mono=True)  # Carregar o áudio do teste
        combined_features = extract_combined_features(audio)  # Extrair características combinadas
        test_features.append(combined_features)  # Adicionar ao conjunto de características do teste
        corrected_true_label = correct_labels(row['Class'])  # Corrigir rótulo se necessário
        true_labels.append(corrected_true_label)  # Adicionar ao conjunto de rótulos do teste
    except Exception as e:
        print(f'Erro ao processar o arquivo {file_name}: {e}')  # Tratar exceções

# Codificar rótulos corrigidos do teste
true_labels_encoded = le.transform(true_labels)

# Fazer previsões com o modelo treinado
predictions = model.predict(np.array(test_features))  # Prever com o modelo

# Calcular rótulos previstos e precisão do conjunto de teste
predicted_labels_encoded = np.argmax(predictions, axis=1)  # Obter os índices máximos
predicted_labels = le.inverse_transform(predicted_labels_encoded)  # Converter para rótulos
accuracy = np.mean(true_labels_encoded == predicted_labels_encoded)  # Calcular precisão

print(f'Precisão da Previsão: {accuracy}')  # Exibir a precisão do teste

# Gerar relatório de classificação
classification_rep = metrics.classification_report(
    true_labels_encoded, predicted_labels_encoded, target_names=le.classes_, digits=4, zero_division=1
)

print("Relatório de Classificação:")  # Exibir o relatório de classificação
print(classification_rep)

# Comparar rótulos verdadeiros e previstos
for true_label, predicted_label in zip(true_labels, predicted_labels):
    print(f'Resposta verdadeira: {true_label}, Predição: {predicted_label}')  # Comparar valores reais e previstos

# Calcular a porcentagem de acertos e a taxa de erro
correct_prediction_percentage = accuracy * 100  # Calcular a porcentagem de predição correta
error_percentage = 100 - correct_prediction_percentage  # Calcular a taxa de erro

print(f"Porcentagem de Predição Correta: {correct_prediction_percentage}%")
print(f"Porcentagem de Erro: {error_percentage}%")

# Gráficos para visualizar a perda durante o treinamento
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Perda durante o Treinamento')  # Título do gráfico
plt.xlabel('Épocas')  # Rótulo do eixo X
plt.ylabel('Perda')  # Rótulo do eixo Y
plt.legend()
plt.grid(True)  # Adicionar grade ao gráfico
plt.show()  # Exibir o gráfico

# Gráfico para visualizar a precisão durante o treinamento
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Precisão de Treinamento')
plt.plot(history.history['val_accuracy'], label='Precisão de Validação')
plt.title('Precisão durante o Treinamento')  # Título do gráfico
plt.xlabel('Épocas')  # Rótulo do eixo X
plt.ylabel('Precisão')  # Rótulo do eixo Y
plt.legend()  # Adicionar legenda
plt.grid(True)  # Adicionar grade ao gráfico
plt.show()  # Exibir o gráfico

confusion_matrix = metrics.confusion_matrix(true_labels_encoded, predicted_labels_encoded)

# Calcular a precisão para cada classe
class_accuracy = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1) * 100

# Plotar a matriz de confusão
plt.figure(figsize=(10, 8))
plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
plt.title('Matriz de Confusão')
plt.colorbar()
tick_marks = np.arange(len(le.classes_))
plt.xticks(tick_marks, le.classes_, rotation=45)
plt.yticks(tick_marks, le.classes_)
plt.xlabel('Rótulos Previstos')
plt.ylabel('Rótulos Verdadeiros')

# Adicionar valores na matriz
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        plt.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='black')

plt.show()

# Exibir as taxas de precisão para cada instrumento
for label, accuracy in zip(le.classes_, class_accuracy):
    print(f'Precisão para {label}: {accuracy:.2f}%')