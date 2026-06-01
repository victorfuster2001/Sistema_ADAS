import os
import sys
import cv2
from ultralytics import YOLO

# 
# CARGA DEL MODELO YOLO
# 

BASE_DIR = getattr(
    sys,
    "_MEIPASS",
    os.path.dirname(os.path.abspath(__file__))
)

modelo_path = os.path.join(BASE_DIR, "yolo11n.pt")

print(f"Cargando modelo YOLOv11 desde: {modelo_path}")

modelo_yolo = YOLO(modelo_path)

# 
# CONFIGURACIÓN CÁMARA
# 

camara = cv2.VideoCapture(1)

if not camara.isOpened():
    raise Exception("No se pudo abrir la cámara")

# 
# PARÁMETROS DISTANCIA
# 

FOCAL = 700

TAMANOS_REALES = {
    "PEATON": 0.5,
    "VEHICULO": 1.8,
    "ANIMAL": 0.7,
    "SENYAL": 0.6,
    "OBSTACULO": 0.5
}

# 
# CLASES SEMÁNTICAS
# 

CLASES_VEHICULO = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle"
}

CLASES_PEATON = {
    "person"
}

CLASES_ANIMAL = {
    "dog",
    "cat",
    "sheep",
    "cow",
    "horse",
    "bird"
}

CLASES_SENAL = {
    "traffic light",
    "stop sign"
}

CLASES_OBSTACULO = {
    "fire hydrant",
    "bench",
    "parking meter"
}

# 
# CLASIFICACIÓN SEMÁNTICA
# 

def clasificar_objeto(nombre_clase):

    if nombre_clase in CLASES_PEATON:
        return "PEATON"

    elif nombre_clase in CLASES_VEHICULO:
        return "VEHICULO"

    elif nombre_clase in CLASES_ANIMAL:
        return "ANIMAL"

    elif nombre_clase in CLASES_SENAL:
        return "SENYAL"

    elif nombre_clase in CLASES_OBSTACULO:
        return "OBSTACULO"

    else:
        return None

# 
# ESTIMACIÓN DISTANCIA
# 

def estimar_distancia(ancho_bbox, ancho_real):

    if ancho_bbox <= 0:
        return None

    distancia = (ancho_real * FOCAL) / ancho_bbox

    return round(distancia, 2)

# 
# LOOP PRINCIPAL
# 

while True:

    valido, frame = camara.read()

    if not valido:
        break

    alto, ancho = frame.shape[:2]

    # 
    # ZONA CENTRAL
    # 

    zona = {
        "x_min": ancho // 3,
        "x_max": 2 * ancho // 3,
        "y_min": alto // 3,
        "y_max": 2 * alto // 3
    }

    cv2.rectangle(
        frame,
        (zona["x_min"], zona["y_min"]),
        (zona["x_max"], zona["y_max"]),
        (255, 0, 0),
        2
    )

    # 
    # YOLO
    # 

    resultados = modelo_yolo.predict(
        frame,
        conf=0.3,
        verbose=False
    )[0]

    # 
    # CONTADORES
    # 

    peatones = 0
    vehiculos = 0
    animales = 0
    senales = 0
    obstaculos = 0

    peatones_en_zona = 0
    vehiculos_en_zona = 0
    animales_en_zona = 0

    # 
    # DISTANCIAS MÍNIMAS
    # 

    distancia_peaton = None
    distancia_vehiculo = None
    distancia_animal = None
    distancia_senal = None
    distancia_obstaculo = None

    # 
    # DETECCIONES
    # 

    for box in resultados.boxes:

        clase_id = int(box.cls[0])

        nombre_clase = resultados.names[clase_id]

        tipo = clasificar_objeto(nombre_clase)

        if tipo is None:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        ancho_bbox = x2 - x1

        centro_x = (x1 + x2) // 2
        centro_y = (y1 + y2) // 2

        # 
        # DISTANCIA
        # 

        ancho_real = TAMANOS_REALES.get(tipo, 1.0)

        distancia = estimar_distancia(
            ancho_bbox,
            ancho_real
        )

        # 
        # PEATÓN
        # 

        if tipo == "PEATON":

            peatones += 1

            color = (0, 255, 0)

            etiqueta = f"PEATON {distancia}m"

            if (
                distancia_peaton is None
                or
                distancia < distancia_peaton
            ):
                distancia_peaton = distancia

            if (
                zona["x_min"] <= centro_x <= zona["x_max"]
                and
                zona["y_min"] <= centro_y <= zona["y_max"]
            ):
                peatones_en_zona += 1

        # 
        # VEHÍCULO
        # 

        elif tipo == "VEHICULO":

            vehiculos += 1

            color = (0, 165, 255)

            etiqueta = f"VEHICULO {distancia}m"

            if (
                distancia_vehiculo is None
                or
                distancia < distancia_vehiculo
            ):
                distancia_vehiculo = distancia

            if (
                zona["x_min"] <= centro_x <= zona["x_max"]
                and
                zona["y_min"] <= centro_y <= zona["y_max"]
            ):
                vehiculos_en_zona += 1

        # 
        # ANIMAL
        # 

        elif tipo == "ANIMAL":

            animales += 1

            color = (0, 0, 255)

            etiqueta = f"ANIMAL {distancia}m"

            if (
                distancia_animal is None
                or
                distancia < distancia_animal
            ):
                distancia_animal = distancia

            if (
                zona["x_min"] <= centro_x <= zona["x_max"]
                and
                zona["y_min"] <= centro_y <= zona["y_max"]
            ):
                animales_en_zona += 1

        # 
        # SENYAL
        # 

        elif tipo == "SENYAL":

            senales += 1

            color = (255, 255, 0)

            etiqueta = f"SENYAL {distancia}m"

            if (
                distancia_senal is None
                or
                distancia < distancia_senal
            ):
                distancia_senal = distancia

        # 
        # OBSTÁCULO
        # 

        elif tipo == "OBSTACULO":

            obstaculos += 1

            color = (255, 0, 255)

            etiqueta = f"OBSTACULO {distancia}m"

            if (
                distancia_obstaculo is None
                or
                distancia < distancia_obstaculo
            ):
                distancia_obstaculo = distancia

        # 
        # DIBUJAR DETECCIÓN
        # 

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.circle(
            frame,
            (centro_x, centro_y),
            4,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            etiqueta,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # 
    # RIESGO
    # 
    distancia_peaton = 1000 if distancia_peaton is None else distancia_peaton
    distancia_vehiculo = 1000 if distancia_vehiculo is None else distancia_vehiculo
    distancia_animal = 1000 if distancia_animal is None else distancia_animal
    distancia_senal = 1000 if distancia_senal is None else distancia_senal
    distancia_obstaculo = 1000 if distancia_obstaculo is None else distancia_obstaculo
    distancia_minima = min(
        distancia_peaton,
        distancia_vehiculo,
        distancia_animal,
        distancia_senal,
        distancia_obstaculo
    )
    riesgo = min(
        100,
        distancia_minima*-5.384+111.912)
    if riesgo < 0: riesgo = 0
    riesgo = int(riesgo)

   
    # 
    # SISTEMA DECISIÓN
    # 

    if riesgo < 20:

        estado = "AVANZAR"

        color_estado = (0, 255, 0)

    elif riesgo < 80:

        estado = "PRECAUCION"

        color_estado = (0, 255, 255)

    else:

        estado = "FRENAR"

        color_estado = (0, 0, 255)

    # 
    # HUD
    # 

    texto_peaton = f"P:{peatones}"
    if distancia_peaton is not None:
        texto_peaton += f" {distancia_peaton}m"

    texto_vehiculo = f"V:{vehiculos}"
    if distancia_vehiculo is not None:
        texto_vehiculo += f" {distancia_vehiculo}m"

    texto_animal = f"A:{animales}"
    if distancia_animal is not None:
        texto_animal += f" {distancia_animal}m"

    texto_senal = f"S:{senales}"
    if distancia_senal is not None:
        texto_senal += f" {distancia_senal}m"

    texto_obstaculo = f"O:{obstaculos}"
    if distancia_obstaculo is not None:
        texto_obstaculo += f" {distancia_obstaculo}m"

    cv2.putText(
        frame,
        texto_peaton,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        texto_vehiculo,
        (180, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 165, 255),
        2
    )

    cv2.putText(
        frame,
        texto_animal,
        (380, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        texto_senal,
        (560, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        texto_obstaculo,
        (720, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"RIESGO: {riesgo}%",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color_estado,
        2
    )

    cv2.putText(
        frame,
        f"ESTADO: {estado}",
        (260, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color_estado,
        2
    )

    # 
    # MOSTRAR FRAME
    # 

    cv2.imshow(
        "Sistema ADAS - Vision Inteligente",
        frame
    )

    # 
    # SALIR
    # 

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 
# LIMPIEZA
# 

camara.release()

cv2.destroyAllWindows()