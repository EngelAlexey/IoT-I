# ================================================================================================
# SmartPill Counter - Conteo de pastillas con Camera Module 3 / v1.3 + Raspberry Pi 4 + 2 Servos SG90
# Curso: ITIEL-11 Internet de las cosas - UTN Pacífico
#
# Hardware : Raspberry Pi 4 + Cámara (puerto CAMERA) + monitor HDMI (micro-HDMI 0) + 2 servos SG90
# Software : Python 3, Picamera2, OpenCV, Flask, SQLite, gpiozero
#
# Uso:
#   python3 contador_pastillas.py
#   - En el monitor de la Raspberry: ventana con el video y el conteo (teclas abajo).
#   - Desde la red: http://<ip de la raspberry>:5000
#
# Teclas en la ventana del monitor:
#   0-9  escribir cantidad objetivo     Enter  confirmar objetivo
#   c    vibrar, capturar y contar      r      volver al conteo en vivo
#   +/-  ajustar umbral manual           a      umbral automático (Otsu)
#   s    guardar registro en SQLite      q      salir
# ================================================================================================

import sqlite3
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

try:
    from picamera2 import Picamera2
    USAR_PICAMERA = True
except ImportError:
    USAR_PICAMERA = False

try:
    from gpiozero import AngularServo
    USAR_SERVO = True
except ImportError:
    USAR_SERVO = False

# ------------------------------------------------------------------------------------------------
# Configuración (calibrar con la bandeja, iluminación y montaje de servos reales)
# ------------------------------------------------------------------------------------------------
RESOLUCION = (1280, 720)
ROI = None                   # Región de la bandeja (x, y, ancho, alto). None = imagen completa
FONDO_OSCURO = True          # True: pastillas claras sobre bandeja oscura/mate
AREA_MIN = 300               # Área mínima en píxeles de una pastilla (descarta polvo y ruido)
AREA_MAX = 20000             # Área máxima de un grupo de pastillas
SEPARACION_WATERSHED = 0.6   # 0.4-0.8: más alto separa más agresivamente pastillas pegadas
LECTURAS_ESTABLES = 5        # Mediana de N cuadros para evitar parpadeo del conteo
DB = "smartpill.db"
MOSTRAR_EN_MONITOR = True    # Ventana OpenCV en el monitor HDMI

PINES_SERVO = [18, 13]       # Servo 1 en GPIO 18 (pin físico 12) y Servo 2 en GPIO 13 (pin físico 33)
CICLOS_VIBRACION = 6         # Cantidad de sacudidas rápidas antes de contar
AMPLITUD_VIBRACION = 15      # Grados de giro a cada lado (ajustar según el soporte)


# ------------------------------------------------------------------------------------------------
# Base de datos (registro de operaciones)
# ------------------------------------------------------------------------------------------------
def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS conteos(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT, objetivo INTEGER, detectadas INTEGER,
            diferencia INTEGER, estado TEXT)""")


def guardar_registro(objetivo, detectadas, estado):
    with sqlite3.connect(DB) as con:
        con.execute("INSERT INTO conteos(fecha, objetivo, detectadas, diferencia, estado) "
                    "VALUES (?,?,?,?,?)",
                    (datetime.now().isoformat(timespec="seconds"), objetivo, detectadas,
                     detectadas - objetivo, estado))


def ultimos_registros(n=20):
    with sqlite3.connect(DB) as con:
        return con.execute("SELECT fecha, objetivo, detectadas, diferencia, estado FROM conteos "
                           "ORDER BY id DESC LIMIT ?", (n,)).fetchall()


# ------------------------------------------------------------------------------------------------
# Cámara
# ------------------------------------------------------------------------------------------------
class Camara:
    def __init__(self):
        if USAR_PICAMERA:
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"size": RESOLUCION, "format": "RGB888"})
            self.picam2.configure(config)
            self.picam2.start()
            try:
                self.picam2.set_controls({"AfMode": 2})
            except Exception as e:
                print(f"Nota sobre autofocus: {e}")
            time.sleep(2)
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUCION[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUCION[1])

    def leer(self):
        if USAR_PICAMERA:
            return self.picam2.capture_array()
        ok, frame = self.cap.read()
        return frame if ok else None

    def cerrar(self):
        if USAR_PICAMERA:
            self.picam2.stop()
        else:
            self.cap.release()


# ------------------------------------------------------------------------------------------------
# Vibración de bandeja con 2 servos SG90
# ------------------------------------------------------------------------------------------------
class VibradorBandeja:
    def __init__(self, pines=PINES_SERVO):
        self.servos = []
        if USAR_SERVO:
            for pin in pines:
                try:
                    s = AngularServo(
                        pin,
                        min_angle=-90,
                        max_angle=90,
                        min_pulse_width=0.0005,
                        max_pulse_width=0.0025
                    )
                    s.angle = 0
                    self.servos.append(s)
                except Exception as e:
                    print(f"Aviso servo en GPIO {pin}: {e}")
            if self.servos:
                time.sleep(0.2)
                for s in self.servos:
                    s.detach()

    def vibrar(self):
        """Sacude la bandeja con ambos servos antes de capturar el conteo."""
        if not self.servos:
            return
        for i in range(CICLOS_VIBRACION):
            sentido = 1 if i % 2 == 0 else -1
            for idx, s in enumerate(self.servos):
                fase = sentido if idx == 0 else -sentido
                s.angle = fase * AMPLITUD_VIBRACION
            time.sleep(0.06)

            for idx, s in enumerate(self.servos):
                fase = -sentido if idx == 0 else sentido
                s.angle = fase * AMPLITUD_VIBRACION
            time.sleep(0.06)

        for s in self.servos:
            s.angle = 0
        time.sleep(0.15)
        for s in self.servos:
            s.detach()
        time.sleep(0.25)

    def cerrar(self):
        for s in self.servos:
            try:
                s.detach()
                s.close()
            except Exception:
                pass


# ------------------------------------------------------------------------------------------------
# Visión: segmentación + watershed para separar pastillas que se tocan
# ------------------------------------------------------------------------------------------------
class Contador:
    def __init__(self):
        self.umbral_manual = None

    def contar(self, frame):
        """Devuelve (cantidad, imagen_anotada, mascara)."""
        x0, y0 = 0, 0
        zona = frame
        if ROI:
            x0, y0, w, h = ROI
            zona = frame[y0:y0 + h, x0:x0 + w]

        gris = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
        gris = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)
        gris = cv2.GaussianBlur(gris, (7, 7), 0)

        tipo = cv2.THRESH_BINARY if FONDO_OSCURO else cv2.THRESH_BINARY_INV
        if self.umbral_manual is None:
            _, mascara = cv2.threshold(gris, 0, 255, tipo + cv2.THRESH_OTSU)
        else:
            _, mascara = cv2.threshold(gris, self.umbral_manual, 255, tipo)

        kernel = np.ones((3, 3), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=2)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)

        fondo_seguro = cv2.dilate(mascara, kernel, iterations=3)
        dist = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
        frente_seguro = np.zeros_like(mascara)
        n_blobs, blobs = cv2.connectedComponents(mascara)
        for b in range(1, n_blobs):
            sel = blobs == b
            local_max = dist[sel].max()
            frente_seguro[sel & (dist > SEPARACION_WATERSHED * local_max)] = 255
        desconocido = cv2.subtract(fondo_seguro, frente_seguro)

        n_marcas, marcadores = cv2.connectedComponents(frente_seguro)
        marcadores = marcadores + 1
        marcadores[desconocido == 255] = 0
        marcadores = cv2.watershed(zona.copy(), marcadores)

        anotada = frame.copy()
        cantidad = 0
        for etiqueta in range(2, n_marcas + 1):
            region = np.uint8(marcadores == etiqueta) * 255
            contornos, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contornos:
                continue
            c = max(contornos, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if AREA_MIN <= area <= AREA_MAX:
                cantidad += 1
                c = c + np.array([x0, y0])
                cv2.drawContours(anotada, [c], -1, (0, 255, 0), 2)
                M = cv2.moments(c)
                if M["m00"]:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    cv2.putText(anotada, str(cantidad), (cx - 8, cy + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if ROI:
            cv2.rectangle(anotada, (x0, y0), (x0 + ROI[2], y0 + ROI[3]), (255, 200, 0), 2)
        return cantidad, anotada, mascara


# ------------------------------------------------------------------------------------------------
# Estado compartido entre la cámara, el monitor y la web
# ------------------------------------------------------------------------------------------------
class Estado:
    def __init__(self):
        self.lock = threading.Lock()
        self.objetivo = 30
        self.detectadas = 0
        self.congelado = False
        self.solicitar_captura = False
        self.jpeg = None
        self.historial = []

    def resultado(self):
        dif = self.detectadas - self.objetivo
        if dif < 0:
            return "INCOMPLETO", f"Faltan {-dif} pastillas", (0, 0, 255)
        if dif > 0:
            return "EXCEDIDO", f"Sobran {dif} pastillas", (0, 165, 255)
        return "COINCIDENTE", "Cantidad correcta", (0, 200, 0)


estado = Estado()
contador = Contador()
vibrador = VibradorBandeja()


def dibujar_panel(img, entrada_objetivo):
    est, msg, color = estado.resultado()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 95), (30, 30, 30), -1)
    cv2.putText(img, f"Objetivo: {estado.objetivo}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, f"Detectadas: {estado.detectadas}", (260, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, f"Diferencia: {estado.detectadas - estado.objetivo:+d}", (560, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, msg, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    modo = "CAPTURA" if estado.congelado else "EN VIVO"
    umbral = "auto" if contador.umbral_manual is None else contador.umbral_manual
    cv2.putText(img, f"{modo} | umbral: {umbral}", (w - 330, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    if entrada_objetivo:
        cv2.putText(img, f"Nuevo objetivo: {entrada_objetivo}_  (Enter)", (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img


def ejecutar_captura_con_vibracion(cam):
    vibrador.vibrar()
    estado.congelado = False
    ultimo_frame = None
    lecturas = []
    for _ in range(LECTURAS_ESTABLES):
        f = cam.leer()
        if f is not None:
            n, anotada, _ = contador.contar(f)
            lecturas.append(n)
            ultimo_frame = anotada
    with estado.lock:
        if lecturas:
            estado.historial = (estado.historial + lecturas)[-LECTURAS_ESTABLES:]
            estado.detectadas = int(np.median(lecturas))
        estado.congelado = True
    return ultimo_frame


def bucle_camara():
    cam = Camara()
    entrada = ""
    ultimo = None
    try:
        while True:
            if estado.solicitar_captura:
                with estado.lock:
                    estado.solicitar_captura = False
                captura = ejecutar_captura_con_vibracion(cam)
                if captura is not None:
                    ultimo = captura

            if not estado.congelado or ultimo is None:
                frame = cam.leer()
                if frame is None:
                    continue
                cantidad, anotada, _ = contador.contar(frame)
                estado.historial = (estado.historial + [cantidad])[-LECTURAS_ESTABLES:]
                with estado.lock:
                    estado.detectadas = int(np.median(estado.historial))
                ultimo = anotada

            vista = dibujar_panel(ultimo.copy(), entrada)
            ok, buf = cv2.imencode(".jpg", vista, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                estado.jpeg = buf.tobytes()

            if MOSTRAR_EN_MONITOR:
                cv2.imshow("SmartPill Counter", vista)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
                elif ord("0") <= tecla <= ord("9"):
                    entrada += chr(tecla)
                elif tecla in (13, 10) and entrada:
                    estado.objetivo = int(entrada)
                    entrada = ""
                elif tecla == 8:
                    entrada = entrada[:-1]
                elif tecla == ord("c"):
                    captura = ejecutar_captura_con_vibracion(cam)
                    if captura is not None:
                        ultimo = captura
                elif tecla == ord("r"):
                    estado.congelado = False
                elif tecla == ord("+"):
                    contador.umbral_manual = min(255, (contador.umbral_manual or 127) + 5)
                elif tecla == ord("-"):
                    contador.umbral_manual = max(0, (contador.umbral_manual or 127) - 5)
                elif tecla == ord("a"):
                    contador.umbral_manual = None
                elif tecla == ord("s"):
                    est, _, _ = estado.resultado()
                    guardar_registro(estado.objetivo, estado.detectadas, est)
                    print(f"Registro guardado: {estado.objetivo} / {estado.detectadas} ({est})")
            else:
                time.sleep(0.03)
    finally:
        vibrador.cerrar()
        cam.cerrar()
        cv2.destroyAllWindows()


# ------------------------------------------------------------------------------------------------
# Interfaz web (componente IoT)
# ------------------------------------------------------------------------------------------------
app = Flask(__name__)

PAGINA = """
<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartPill Counter</title>
<style>
 body{font-family:Arial,sans-serif;background:#1e1e2e;color:#cdd6f4;text-align:center;margin:0;padding:20px}
 h1{color:#89b4fa} img{border:3px solid #89b4fa;border-radius:10px;max-width:100%}
 .datos{display:flex;gap:20px;justify-content:center;margin:15px 0;font-size:1.4em}
 .datos div{background:#313244;padding:10px 20px;border-radius:8px}
 #msg{font-size:1.6em;font-weight:bold} button,input{font-size:1em;padding:6px 12px;margin:4px}
 table{margin:20px auto;border-collapse:collapse} td,th{border:1px solid #45475a;padding:4px 10px}
</style></head><body>
<h1>SmartPill Counter</h1>
<img src="/video_feed" alt="Cámara">
<div class="datos">
 <div>Objetivo: <b id="obj">-</b></div><div>Detectadas: <b id="det">-</b></div>
 <div>Diferencia: <b id="dif">-</b></div></div>
<div id="msg">-</div>
<p><input id="nuevo" type="number" min="1" placeholder="Cantidad objetivo">
<button onclick="fijar()">Fijar objetivo</button>
<button onclick="post('/capturar')">Capturar</button>
<button onclick="post('/vivo')">En vivo</button>
<button onclick="post('/guardar')">Guardar registro</button></p>
<table id="tabla"><tr><th>Fecha</th><th>Objetivo</th><th>Detectadas</th><th>Dif.</th><th>Estado</th></tr></table>
<script>
const colores={INCOMPLETO:'#f38ba8',EXCEDIDO:'#fab387',COINCIDENTE:'#a6e3a1'};
async function actualizar(){
 const r=await (await fetch('/estado')).json();
 obj.textContent=r.objetivo; det.textContent=r.detectadas;
 dif.textContent=(r.diferencia>0?'+':'')+r.diferencia;
 msg.textContent=r.mensaje; msg.style.color=colores[r.estado];
}
async function registros(){
 const r=await (await fetch('/registros')).json();
 tabla.innerHTML='<tr><th>Fecha</th><th>Objetivo</th><th>Detectadas</th><th>Dif.</th><th>Estado</th></tr>'+
  r.map(x=>'<tr>'+x.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('');
}
async function post(url,body){await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify(body||{})}); actualizar(); registros();}
function fijar(){const v=parseInt(nuevo.value); if(v>0) post('/objetivo',{objetivo:v});}
setInterval(actualizar,500); actualizar(); registros();
</script></body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGINA)


@app.route("/video_feed")
def video_feed():
    def generar():
        while True:
            if estado.jpeg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + estado.jpeg + b"\r\n")
            time.sleep(0.05)
    return Response(generar(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/estado")
def api_estado():
    est, msg, _ = estado.resultado()
    return jsonify(objetivo=estado.objetivo, detectadas=estado.detectadas,
                   diferencia=estado.detectadas - estado.objetivo,
                   estado=est, mensaje=msg, congelado=estado.congelado)


@app.route("/objetivo", methods=["POST"])
def api_objetivo():
    estado.objetivo = int(request.json.get("objetivo", estado.objetivo))
    return jsonify(ok=True, objetivo=estado.objetivo)


@app.route("/capturar", methods=["POST"])
def api_capturar():
    with estado.lock:
        estado.solicitar_captura = True
    time.sleep(1.3)
    return jsonify(ok=True, detectadas=estado.detectadas)


@app.route("/vivo", methods=["POST"])
def api_vivo():
    estado.congelado = False
    return jsonify(ok=True)


@app.route("/guardar", methods=["POST"])
def api_guardar():
    est, _, _ = estado.resultado()
    guardar_registro(estado.objetivo, estado.detectadas, est)
    return jsonify(ok=True)


@app.route("/registros")
def api_registros():
    return jsonify(ultimos_registros())


if __name__ == "__main__":
    init_db()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False,
                                            use_reloader=False), daemon=True).start()
    bucle_camara()
    
