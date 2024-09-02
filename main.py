import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD

# Función para leer un archivo CSV y cargarlo en un DataFrame
def cargar_datos_csv(archivo):
    try:
        df = pd.read_csv(archivo)
        return df
    except FileNotFoundError:
        print(f'No se encontró el archivo {archivo}')
        return None

# Función para normalizar las columnas 'Altura' y 'Peso' en el DataFrame
def normalizar_columnas(df):
    df['Altura'] = df['Altura'] / df['Altura'].max()
    df['Peso'] = df['Peso'] / df['Peso'].max() 
    return df

# Función para crear y configurar un modelo de regresión lineal simple
def crear_modelo():
    np.random.seed(2)  # Fijar la semilla para reproducibilidad
    modelo = Sequential()  # Crear un modelo secuencial

    # Definir las dimensiones de entrada y salida
    dimension_entrada = 1  
    dimension_salida = 1
    
    # Añadir una capa densa con activación lineal
    capa = Dense(dimension_salida, input_dim=dimension_entrada, activation='linear')
    modelo.add(capa)
    
    # Configurar el optimizador SGD (Descenso de Gradiente Estocástico)
    optimizador_sgd = SGD(learning_rate=0.0004)
    modelo.compile(loss='mse', optimizer=optimizador_sgd)  # Compilar el modelo con pérdida MSE (Error Cuadrático Medio)
    modelo.summary()  # Mostrar un resumen del modelo
    return modelo

# Función para entrenar el modelo con los datos
def entrenar_modelo(modelo, df):
    # Extraer las características (x) y la variable objetivo (y)
    x = df['Altura'].values
    y = df['Peso'].values
    
    num_epocas = 10000  # Número de épocas de entrenamiento
    tamano_lote = x.shape[0]  # Usar todos los datos en cada lote
    
    # Entrenar el modelo
    historial = modelo.fit(x, y, epochs=num_epocas, batch_size=tamano_lote, verbose=1)
    
    # Obtener los pesos de la primera capa
    capa = modelo.layers[0]
    w, b = capa.get_weights()  # w = peso, b = sesgo
    print('Parámetros: w = {:.4f}, b = {:.4f}'.format(w[0][0], b[0]))
    
    return historial, w[0][0], b[0]

# Función para graficar el Error Cuadrático Medio (ECM) a lo largo de las épocas
def graficar_ecm(historial):
    plt.figure(figsize=(10, 5))
    plt.plot(historial.history['loss'], 'b-', label='ECM')
    plt.xlabel('Épocas')
    plt.ylabel('ECM')
    plt.title('ECM vs. Épocas')
    plt.legend()
    plt.grid(True)
    plt.show()

# Función para graficar la recta de regresión y los datos originales
def graficar_regresion(df, w, b):
    x = df['Altura'].values
    y = df['Peso'].values
    
    # Calcular la predicción usando la recta de regresión
    y_prediccion = w * x + b
    
    plt.scatter(x, y, label='Datos originales', color='blue')
    plt.plot(x, y_prediccion, label='Recta de Regresión', color='red')
    plt.xlabel('Altura (Normalizada)')
    plt.ylabel('Peso (Normalizado)')
    plt.title('Recta de Regresión vs. Datos Originales')
    plt.legend()
    plt.show()

# Función para hacer predicciones con el modelo entrenado
def predecir_peso(modelo, altura_cm, df):
    # Normalizar la altura dada
    altura_maxima = df['Altura'].max()
    altura_normalizada = altura_cm / altura_maxima
    
    # Realizar la predicción
    prediccion_y = modelo.predict(np.array([altura_normalizada]))
    
    # Escalar la predicción de nuevo a la escala original
    peso_maximo = df['Peso'].max()
    peso = prediccion_y[0][0] * peso_maximo
    
    print(f'El peso para una persona de {altura_cm} cm es de {peso:.2f} kg')
    return peso

# Función principal que ejecuta el flujo del programa
def main():
    df = cargar_datos_csv('altura_peso.csv')
    df = normalizar_columnas(df)
    modelo = crear_modelo()
    historial, w, b = entrenar_modelo(modelo, df)
    graficar_ecm(historial)
    graficar_regresion(df, w, b)
    
    # Predecir el peso para una altura dada
    altura = 170
    predecir_peso(modelo, altura, df)
    
if __name__ == '__main__':
    main()
