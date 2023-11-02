import os
import time
import threading
from gtts import gTTS
from playsound import playsound
from ultralytics import YOLO

# Inicializa el modelo YOLO
print("Cargando modelo YOLO...")
model = YOLO("yolov8x.pt")
print("Modelo cargado.")

# Función para reproducir los nombres de los objetos detectados
def speak_objects(names):
    if names:  # Solo procede si hay algo que decir
        text = ', '.join(names)
        print(f"Preparando para hablar: {text}")
        tts = gTTS(text=text, lang='en')
        filename = 'temp.mp3'
        tts.save(filename)
        print("Reproduciendo sonido...")
        playsound(filename)
        os.remove(filename)  # Elimina el archivo temporal después de reproducirlo
        print("Sonido reproducido y archivo temporal eliminado.")

# Función de trabajo de detección
def detection_worker():
    print("Iniciando detección...")
    # Utiliza el modo de streaming con visualización
    results = model.predict(source="0", stream=True, show=True)  # Habilita la visualización de la cámara
    for result in results:
        start_time = time.time()  # Registra el tiempo de inicio
        print("Detectando...")

        # Intenta acceder a las propiedades de 'result'
        try:
            if result.boxes.shape[0] > 0:  # Verifica que haya detecciones
                # Recupera los nombres de las clases detectadas utilizando el atributo 'cls'
                names = [model.names[int(cls)] for cls in result.boxes.cls]
                print(f"Objetos detectados: {names}")
                # Usa un hilo para hablar de los objetos detectados
                speak_thread = threading.Thread(target=speak_objects, args=(names,))
                speak_thread.start()
                speak_thread.join()  # Espera a que el hilo de voz termine antes de continuar
        except Exception as e:
            print(f"Error al procesar los resultados: {e}")

        # Calcular cuánto tiempo esperar teniendo en cuenta el tiempo de detección
        detection_time = time.time() - start_time
        wait_time = max(5 - detection_time, 0)
        print(f"Esperando {wait_time:.2f} segundos para la siguiente detección.")
        time.sleep(wait_time)  # Espera hasta completar 5 segundos

# Iniciar el hilo de detección
print("Iniciando el hilo de detección...")
detection_thread = threading.Thread(target=detection_worker)
detection_thread.start()
