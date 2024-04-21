import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import librosa  # Biblioteca para processamento de áudio
from sklearn.model_selection import train_test_split  # Divisão entre treino e validação
from sklearn.preprocessing import LabelEncoder  # Codificação de rótulos
import sklearn.metrics as metrics  # Métricas para relatórios de classificação

# Definir os caminhos para os dados de áudio e metadados
base_dir = 'archive/'  # Diretório base para os dados
train_audio_dir = os.path.join(base_dir, 'Train_submission/Train_submission/')  # Diretório para áudio de treinamento
metadata_train_path = os.path.join(base_dir, 'Metadata_Train.csv')  # Caminho para os metadados de treinamento

# Carregar os metadados de treinamento e o modelo YAMNet
metadata_train = pd.read_csv(metadata_train_path)  # Carregar metadados de treino de um arquivo CSV
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'  # URL para o modelo YAMNet
yamnet_model = hub.load(yamnet_model_handle)  # Carregar o modelo do TensorFlow Hub

# Funções de aumento de dados (data augmentation)
def add_noise(data, noise_factor=0.02):
    # Adiciona ruído ao sinal de áudio para aumentar a diversidade
    noise = noise_factor * np.random.randn(len(data))
    return data + noise  # Retorna o áudio com ruído adicionado

def change_speed(data, speed_factor=0.8):
    # Altera a velocidade do sinal de áudio
    return librosa.effects.time_stretch(y=data, rate=speed_factor)  # Aplicar efeito de mudança de velocidade

def change_pitch(data, pitch_steps=4):
    # Altera a tonalidade do sinal de áudio
    return librosa.effects.pitch_shift(data, sr=16000, n_steps=pitch_steps)  # Deslocamento de semitons para mudança de pitch

def add_echo(data, delay=0.2, decay=0.5, sr=16000):
    # Adiciona eco ao sinal de áudio
    delay_samples = int(delay * sr)  # Converte o atraso para amostras
    echo = np.zeros(len(data) + delay_samples)  # Cria um sinal com espaço para o eco
    echo[:len(data)] = data  # Adiciona o áudio original ao início do sinal
    echo[delay_samples:] += data * decay  # Adiciona o eco com decaimento
    return echo  # Retorna o áudio com eco

def modulate_amplitude(data, modulation_frequency=1.0, sr=16000):
    # Modula a amplitude do sinal para introduzir variações
    time = np.linspace(0, len(data) / sr, len(data))
    modulation = 1.0 + 0.5 * np.sin(2 * np.pi * modulation_frequency * time)  # Criar onda seno para modulação
    return data * modulation  # Retorna o áudio modulado

def change_volume(data, volume_factor=0.8):
    # Multiplica o sinal por um fator para ajustar o volume
    return data * volume_factor  # Ajuste do volume

def transpose_frequency(data, shift_semitones=2):
    # Transpõe a frequência do sinal de áudio para variar a tonalidade
    return librosa.effects.pitch_shift(data, sr=16000, n_steps=shift_semitones)

def reverse_audio(data):
    # Inverte o sinal de áudio
    return data[::-1]  # Retorna o áudio invertido

def shift_phase(data, phase_shift=0.1):
    # Altera a fase do sinal
    phase = np.angle(np.fft.fft(data))  # Obter a fase com FFT
    magnitude = np.abs(np.fft.fft(data))  # Obter a magnitude do FFT
    shifted = np.fft.ifft(magnitude * np.exp(1j * (phase + phase_shift)))  # Aplicar mudança de fase
    return shifted.real  # Retorna a parte real do sinal

# Função para extrair características do áudio
def extract_features(file_path, model, augment=True):
    # Carregar o arquivo de áudio com uma frequência de amostragem de 16 kHz e em mono
    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    # Aplicar aumento de dados com base em condições aleatórias para introduzir diversidade
    if augment:
        if np.random.rand() < 0.5:
            volume_factor = np.random.uniform(0.7, 1.3)  # Fator aleatório para ajuste de volume
            audio = change_volume(audio, volume_factor)  # Ajuste do volume
        if np.random.rand() < 0.3:
            audio = add_echo(audio, delay=0.1, decay=0.6, sr=16000)  # Adicionar eco
        if np.random.rand() < 0.3:
            audio = modulate_amplitude(audio, modulation_frequency=0.5, sr=16000)  # Modulação de amplitude
        if np.random.rand() < 0.3:
            audio = reverse_audio(audio)  # Inverter o áudio
        if np.random.rand() < 0.5:
            audio = add_noise(audio, noise_factor=0.03)  # Adicionar ruído
        if np.random.rand() < 0.5:
            audio = change_speed(audio, speed_factor=0.9)  # Mudança de velocidade
        if np.random.rand() < 0.5:
            audio = change_pitch(audio, pitch_steps=2)  # Menos mudança de pitch

    # Ajustar o comprimento do áudio para um tamanho fixo para garantir consistência
    if len(audio) > 160000:
        audio = audio[:160000]  # Cortar excesso se necessário
    elif len(audio) < 160000:
        padding = 160000 - len(audio)  # Adicionar preenchimento para completar o comprimento
        audio = np.pad(audio, (0, padding), 'constant')

    # Converter o áudio para tensor do TensorFlow
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)  # Conversão para tensor
    waveform = tf.reshape(waveform, [-1])  # Ajuste para formato de onda

    # Obter embeddings do modelo YAMNet para uso no treinamento
    _, embeddings, _ = model(waveform)  # Obter embeddings

    return tf.reduce_mean(embeddings, axis=0).numpy()  # Retornar a média dos embeddings

# Extração de características e codificação de rótulos para o conjunto de treinamento
features = []  # Lista para armazenar características extraídas
labels = []  # Lista para rótulos dos arquivos de áudio

# Iterar sobre os metadados para processar cada arquivo de áudio
for index, row in metadata_train.iterrows():
    file_name = row['FileName']  # Nome do arquivo de áudio
    file_path = os.path.join(train_audio_dir, file_name)  # Caminho completo para o arquivo
    try:
        # Extrair características do áudio e armazenar na lista
        print("Processando arquivo de treino:", file_name)
        embedding = extract_features(file_path, yamnet_model)  # Obter as características
        features.append(embedding)  # Adicionar à lista de características
        labels.append(row['Class'])  # Adicionar rótulo à lista
    except Exception as e:
        print(f'Erro ao processar o arquivo {file_name}: {e}')  # Relatar erro, se houver

# Codificação de rótulos para treinamento do modelo
le = LabelEncoder()  # Codificador de rótulos para conversão de classes
labels_encoded = le.fit_transform(labels)  # Codificar rótulos em formato numérico
labels_onehot = tf.keras.utils.to_categorical(labels_encoded)  # Codificação one-hot

# Divisão de treino e validação
X_train, X_val, y_train, y_val = train_test_split(
    np.array(features), np.array(labels_onehot), test_size=0.2, random_state=42
)  # Divisão dos dados em 80% treino e 20% validação

# Configuração do modelo de aprendizado de máquina
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1024,), name='input_embedding'),  # Camada de entrada para embeddings
    tf.keras.layers.Dense(128, activation='relu'),  # Camada densa com ReLU e 128 neurônios
    tf.keras.layers.Dropout(0.4),  # Dropout para regularização
    tf.keras.layers.Dense(64, activation='relu'),  # Camada densa adicional
    tf.keras.layers.Dropout(0.2),  # Dropout adicional
    tf.keras.layers.Dense(len(np.unique(labels_encoded)), activation='softmax')  # Saída softmax
])

# Configuração do otimizador para o modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Definir a taxa de aprendizado para Adam
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callback para parada precoce para evitar overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True  # Parada precoce após 10 épocas sem melhora
)

# Treinamento do modelo com validação
history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[early_stopping]  # Treinamento com validação
)

# Avaliação do modelo após treinamento
val_loss, val_accuracy = model.evaluate(X_val, y_val)  # Obter perda e precisão do conjunto de validação
print(f'Perda de Validação: {val_loss}')  # Mostrar perda de validação
print(f'Precisão de Validação: {val_accuracy}')  # Mostrar precisão de validação

# Extração de características para o conjunto de teste e codificação de rótulos
metadata_test_path = os.path.join(base_dir, 'Metadata_Test.csv')  # Diretório para metadados de teste
test_audio_dir = os.path.join(base_dir, 'Test_submission/Test_submission/') # Diretório para áudio de treinamento
metadata_test = pd.read_csv(metadata_test_path)  # Carregar metadados de teste

test_features = []  # Lista para armazenar características do teste
true_labels = []  # Lista para rótulos verdadeiros do teste

# Iterar sobre os metadados de teste para extrair características
for index, row in metadata_test.iterrows():
    file_name = row['FileName']  # Nome do arquivo de teste
    file_path = os.path.join(test_audio_dir, file_name)  # Caminho para o arquivo
    try:
        # Extrair características do áudio para o conjunto de teste
        print("Processando arquivo de teste:", file_name)
        embedding = extract_features(file_path, yamnet_model)  # Obter características do YAMNet
        test_features.append(embedding)  # Adicionar à lista de características do teste
        true_label = row['Class']  # Obter rótulo verdadeiro
        # Corrigir rótulo se necessário
        if true_label == "Sound_Guiatr":
            true_label = "Sound_Guitar"  # Corrigir erro de digitação
        true_labels.append(true_label)  # Adicionar rótulo corrigido
    except Exception as e:
        print(f'Erro ao processar o arquivo {file_name}: {e}')  # Lidar com exceções durante a extração

# Codificar rótulos para o conjunto de teste
true_labels_encoded = le.transform(true_labels)  # Transformar rótulos para formato numérico

# Previsões do modelo para o conjunto de teste
predictions = model.predict(np.array(test_features))  # Fazer previsões para o teste

# Determinar rótulos previstos a partir do índice máximo
predicted_labels_encoded = np.argmax(predictions, axis=1)  # Obter índices máximos das previsões
predicted_labels = le.inverse_transform(predicted_labels_encoded)  # Converter para rótulos legíveis

# Calcular precisão geral do modelo
accuracy = np.mean(true_labels_encoded == predicted_labels_encoded)  # Calcular precisão

print(f'Precisão da Previsão: {accuracy}')  # Mostrar precisão geral do teste

# Relatório de classificação detalhado
classification_rep = metrics.classification_report(
    true_labels_encoded, predicted_labels_encoded, target_names=le.classes_, digits=4, zero_division=1
)  # Relatório detalhado com precisão, recall e f1-score

print("Relatório de Classificação:")
print(classification_rep)  # Mostrar o relatório detalhado

# Comparação entre rótulos verdadeiros e previstos
for true_label, predicted_label in zip(true_labels, predicted_labels):
    print(f'Resposta verdadeira: {true_label}, Predição: {predicted_label}')  # Mostrar comparação entre rótulos

# Cálculo da porcentagem de predição correta e da taxa de erro
correct_prediction_percentage = accuracy * 100  # Converter precisão para porcentagem
error_percentage = 100 - correct_prediction_percentage  # Calcular taxa de erro

print(f"Porcentagem de Predição Correta: {correct_prediction_percentage}%")
print(f"Porcentagem de Erro: {error_percentage}%")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Perda durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.grid(True)
plt.show()

# Plotar a precisão durante o treinamento
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Precisão de Treinamento')
plt.plot(history.history['val_accuracy'], label='Precisão de Validação')
plt.title('Precisão durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Precisão')
plt.legend()
plt.grid(True)
plt.show()