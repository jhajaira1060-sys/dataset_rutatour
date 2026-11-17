import io

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

# Permite que tu front (Angular en localhost:4200) llame a esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Se carga UNA sola vez al arrancar la API
RUTA_MODELO = "runs/detect/train/weights/best.pt"
modelo = YOLO(RUTA_MODELO)

CONFIANZA_MINIMA = 0.4


def predecir(archivo: UploadFile):
    imagen = Image.open(archivo.file).convert("RGB")
    return modelo.predict(imagen, conf=CONFIANZA_MINIMA, verbose=False)[0]


@app.get("/salud")
def salud():
    return {"estado": "ok", "clases": modelo.names}


@app.post("/detectar")
def detectar(imagen: UploadFile = File(...)):
    resultado = predecir(imagen)

    detecciones = []
    for caja in resultado.boxes:
        x1, y1, x2, y2 = caja.xyxy[0].tolist()
        detecciones.append({
            "clase": modelo.names[int(caja.cls)],
            "confianza": round(float(caja.conf), 3),
            "caja": [round(x1), round(y1), round(x2), round(y2)],
        })

    return {
        "total": len(detecciones),
        "ancho": resultado.orig_shape[1],
        "alto": resultado.orig_shape[0],
        "detecciones": detecciones,
    }


@app.post("/detectar-imagen")
def detectar_imagen(imagen: UploadFile = File(...)):
    """Devuelve la misma foto con las cajas dibujadas (útil para probar)."""
    resultado = predecir(imagen)
    dibujada = resultado.plot()[:, :, ::-1]  # BGR -> RGB
    buffer = io.BytesIO()
    Image.fromarray(dibujada).save(buffer, format="JPEG")
    return Response(content=buffer.getvalue(), media_type="image/jpeg")