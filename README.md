# README.md

## Sistema ADAS con Visión Artificial en Tiempo Real

Este proyecto implementa un sistema básico de asistencia a la conducción (ADAS) utilizando visión artificial e inteligencia artificial en tiempo real. El sistema emplea el modelo YOLOv11 para detectar distintos elementos presentes en la carretera mediante una cámara conectada al ordenador.

Entre las funcionalidades principales se incluyen:

* Detección de peatones
* Detección de vehículos
* Detección de animales
* Detección de señales y obstáculos
* Estimación aproximada de distancias
* Cálculo dinámico del nivel de riesgo
* Generación de estados de conducción:

  * AVANZAR
  * PRECAUCIÓN
  * FRENAR

Toda la información se muestra directamente sobre la imagen capturada por la cámara en tiempo real.

---

# Requisitos

El proyecto ha sido desarrollado en Python y requiere las siguientes librerías:

```bash
pip install ultralytics opencv-python matplotlib
```

También es necesario disponer del modelo:

```text
yolo11n.pt
```

El archivo debe encontrarse en la misma carpeta que el programa principal.

---

# Ejecución del programa

Para ejecutar el sistema:

```bash
python main.py
```

---

# Configuración de cámara

El sistema está configurado por defecto para utilizar una segunda cámara externa mediante la siguiente línea:

```python
camara = cv2.VideoCapture(1)
```

Si se desea utilizar la cámara integrada del portátil, es necesario modificar el valor `1` por `0`:

```python
camara = cv2.VideoCapture(0)
```

---

# Funcionamiento general

El programa procesa continuamente los frames capturados por la cámara y ejecuta los siguientes pasos:

1. Captura de imagen en tiempo real
2. Detección de objetos mediante YOLOv11
3. Clasificación semántica de objetos
4. Estimación de distancias aproximadas
5. Cálculo del nivel de riesgo
6. Generación del estado de conducción
7. Representación visual de resultados

Además, el sistema dibuja:

* Bounding boxes
* Distancias estimadas
* Número de objetos detectados
* Nivel de riesgo
* Estado del sistema

---

# Estimación de distancias

El cálculo de distancia se realiza mediante visión monocular utilizando la fórmula:

```text
distancia = (ancho_real × focal) / ancho_bbox
```

Este método proporciona una aproximación suficiente para entornos académicos y prototipos de conducción asistida.

---

# Estados del sistema

El sistema genera tres niveles de decisión:

| Riesgo | Estado     |
| ------ | ---------- |
| Bajo   | AVANZAR    |
| Medio  | PRECAUCIÓN |
| Alto   | FRENAR     |

---

# Salida del programa

Para cerrar correctamente el sistema y finalizar la ejecución, es necesario pulsar la tecla:

```text
q
```

---

# Estructura del proyecto

```text
Proyecto/
│
├── main.py
├── yolo11n.pt
└── README.md
```

---

# Tecnologías utilizadas

* Python
* OpenCV
* Ultralytics YOLOv11
* Visión artificial
* Inteligencia artificial aplicada a ADAS

---

# Autores

Víctor Fuster
