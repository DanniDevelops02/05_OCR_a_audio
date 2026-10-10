import base64
import io
import re
import time
from html import escape
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import streamlit as st
import streamlit.components.v1 as components
from deep_translator import GoogleTranslator, MyMemoryTranslator
from gtts import gTTS
from PIL import Image

st.set_page_config(
    page_title="Imagen a audio · Daniel Flórez",
    page_icon="🔊",
    layout="centered",
)

# =====================================================================
#  CONTENIDO
# =====================================================================
IMAGEN = "ocr_audio.jpg"  # opcional: si no existe, simplemente no se muestra

# codigo   = idioma para traducir y generar la voz
# respaldo = código del traductor de respaldo (MyMemory)
# ocr      = idioma que Tesseract debe buscar en la imagen
# paquete  = paquete de Linux que instala ese idioma de OCR
IDIOMAS = {
    "Español": {"codigo": "es", "respaldo": "es-ES", "ocr": "spa", "paquete": "tesseract-ocr-spa"},
    "Inglés": {"codigo": "en", "respaldo": "en-GB", "ocr": "eng", "paquete": "tesseract-ocr"},
    "Bengalí": {"codigo": "bn", "respaldo": "bn-IN", "ocr": "ben", "paquete": "tesseract-ocr-ben"},
    "Coreano": {"codigo": "ko", "respaldo": "ko-KR", "ocr": "kor", "paquete": "tesseract-ocr-kor"},
    "Mandarín": {"codigo": "zh-CN", "respaldo": "zh-CN", "ocr": "chi_sim", "paquete": "tesseract-ocr-chi-sim"},
    "Japonés": {"codigo": "ja", "respaldo": "ja-JP", "ocr": "jpn", "paquete": "tesseract-ocr-jpn"},
}

ACENTOS = {
    "Defecto": "com",
    "México": "com.mx",
    "India": "co.in",
    "Reino Unido": "co.uk",
    "Estados Unidos": "com",
    "Canadá": "ca",
    "Australia": "com.au",
    "Irlanda": "ie",
    "Sudáfrica": "co.za",
}

FILTROS = {
    "Escala de grises": "Convierte a blanco y negro suave. Es la mejor opción para empezar.",
    "Alto contraste": "Resalta el texto frente al fondo. Útil con sombras o poca luz.",
    "Invertir colores": "Para texto claro sobre fondo oscuro.",
    "Sin filtro": "Usa la imagen tal como la captura la cámara.",
}

# =====================================================================
#  ESTILO (paleta fría basada en #3f71a6)
# =====================================================================
PRIMARIO = "#3f71a6"
OSCURO = "#1b2f45"
CLARO = "#eaf2f9"
SUAVE = "#a9c7e3"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    font-size: 1.08rem;
}
.stApp { background: linear-gradient(180deg, #f6fafd 0%, __CLARO__ 100%); }
.block-container { max-width: 860px; padding-top: 3rem; padding-bottom: 4rem; }

/* Hero */
.hero { text-align: center; padding: 1rem 0 .5rem 0; }
.hero .etiqueta {
    color: __PRIMARIO__; letter-spacing: .25em; text-transform: uppercase;
    font-size: .85rem; font-weight: 600;
}
.hero h1 {
    font-family: 'Playfair Display', serif; font-size: 3.2rem; color: __OSCURO__;
    margin: .3rem 0 .6rem 0; line-height: 1.1;
}
.hero p { color: #4d6479; font-size: 1.2rem; max-width: 640px; margin: 0 auto; }
.linea {
    width: 70px; height: 4px; border-radius: 4px; background: __PRIMARIO__;
    margin: 1.4rem auto;
}
h2, h3 { font-family: 'Playfair Display', serif !important; color: __OSCURO__ !important; }

/* Tarjetas */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border: 1px solid __SUAVE__66 !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 24px rgba(63, 113, 166, .08);
    padding: .6rem .8rem;
}

/* Imágenes (sin recortar: en OCR importa ver la foto completa) */
[data-testid="stImage"] img {
    border-radius: 14px;
    box-shadow: 0 8px 22px rgba(27, 47, 69, .15);
}
[data-testid="stImageCaption"] { text-align: center; color: #6b8399; }

/* Cámara y subida de archivos */
[data-testid="stCameraInput"] > div, [data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img { border-radius: 16px; }
[data-testid="stFileUploaderDropzone"] {
    background: __CLARO__; border: 2px dashed __SUAVE__; border-radius: 16px;
}

/* Resultado de la traducción */
.resultado {
    font-family: 'Playfair Display', serif; font-size: 1.4rem; line-height: 1.6;
    color: __OSCURO__; padding: .4rem .2rem 1rem .2rem; white-space: pre-wrap;
}

/* Botones */
.stButton > button, .stDownloadButton > button {
    background: __PRIMARIO__; color: white; border: 1.5px solid __PRIMARIO__;
    border-radius: 999px; padding: .7rem 1.6rem; font-size: 1.02rem; font-weight: 600;
    transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: __OSCURO__; border-color: __OSCURO__; color: white;
    transform: translateY(-2px); box-shadow: 0 8px 18px rgba(27, 47, 69, .25);
}
[data-testid="stBaseButton-secondary"] { background: white; color: __PRIMARIO__; }

/* Inputs */
.stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
    border-radius: 12px; border: 1px solid __SUAVE__; font-size: 1.05rem;
}
[data-testid="stAlert"] { border-radius: 14px; }

/* Sidebar */
[data-testid="stSidebar"] { background: __OSCURO__; }
[data-testid="stSidebar"] * { color: #e6f0f9 !important; }

#MainMenu, footer { visibility: hidden; }
</style>
"""

for token, valor in {
    "__PRIMARIO__": PRIMARIO,
    "__OSCURO__": OSCURO,
    "__CLARO__": CLARO,
    "__SUAVE__": SUAVE,
}.items():
    CSS = CSS.replace(token, valor)

st.markdown(CSS, unsafe_allow_html=True)

# =====================================================================
#  REPRODUCTOR CON ONDA DE SONIDO (el mismo de las apps de audio)
# =====================================================================
PLAYER_HTML = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Inter', sans-serif; background: transparent; }
  .player { display: flex; align-items: center; gap: 18px; padding: 8px 2px; }
  .play {
    width: 66px; height: 66px; border-radius: 50%; border: none; flex: none;
    background: __PRIMARIO__; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background .2s ease, transform .2s ease;
  }
  .play:hover { background: __OSCURO__; transform: scale(1.06); }
  .play svg { width: 28px; height: 28px; fill: white; }
  .wave { flex: 1; height: 88px; display: flex; align-items: center; gap: 3px; cursor: pointer; }
  .bar { flex: 1; min-width: 2px; border-radius: 3px; background: __SUAVE__; transition: background .15s; }
  .bar.done { background: __PRIMARIO__; }
  .time { min-width: 92px; text-align: right; font-size: .9rem; color: #4d6479;
          font-variant-numeric: tabular-nums; }
</style>

<div class="player">
  <button class="play" id="btn" aria-label="Reproducir o pausar">
    <svg viewBox="0 0 24 24"><path id="icon" d="M8 5v14l11-7z"/></svg>
  </button>
  <div class="wave" id="wave" title="Haz clic para saltar a otro punto"></div>
  <div class="time" id="time">0:00 / 0:00</div>
</div>
<audio id="audio" preload="auto" src="data:audio/mpeg;base64,__AUDIO__"></audio>

<script>
  const N = 56;
  const PLAY = "M8 5v14l11-7z";
  const PAUSE = "M6 5h4v14H6zM14 5h4v14h-4z";
  const audio = document.getElementById("audio");
  const wave = document.getElementById("wave");
  const btn = document.getElementById("btn");
  const icon = document.getElementById("icon");
  const time = document.getElementById("time");

  const base = Array.from({length: N}, (_, i) =>
    0.22 + 0.78 * Math.abs(Math.sin(i * 0.9) * Math.cos(i * 0.37)));
  const bars = base.map(() => {
    const d = document.createElement("div");
    d.className = "bar";
    wave.appendChild(d);
    return d;
  });

  let playing = false;

  function fmt(s) {
    if (!isFinite(s)) return "0:00";
    const m = Math.floor(s / 60), r = Math.floor(s % 60);
    return m + ":" + String(r).padStart(2, "0");
  }

  function draw(t) {
    const dur = isFinite(audio.duration) ? audio.duration : 0;
    const p = dur ? audio.currentTime / dur : 0;
    bars.forEach((b, i) => {
      const wob = playing ? 0.68 + 0.32 * Math.sin(t / 130 + i * 0.7) : 1;
      b.style.height = (base[i] * wob * 100) + "%";
      b.classList.toggle("done", i / N < p);
    });
    time.textContent = fmt(audio.currentTime) + " / " + fmt(dur);
    if (playing) requestAnimationFrame(draw);
  }

  btn.onclick = () => (audio.paused ? audio.play() : audio.pause());

  audio.addEventListener("play", () => {
    playing = true;
    icon.setAttribute("d", PAUSE);
    requestAnimationFrame(draw);
  });
  const detener = () => {
    playing = false;
    icon.setAttribute("d", PLAY);
    draw(0);
  };
  audio.addEventListener("pause", detener);
  audio.addEventListener("ended", detener);
  audio.addEventListener("loadedmetadata", () => draw(0));

  wave.addEventListener("click", (e) => {
    const r = wave.getBoundingClientRect();
    if (isFinite(audio.duration)) {
      audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration;
      if (!playing) draw(0);
    }
  });

  draw(0);
</script>
"""


def reproductor(audio_bytes: bytes) -> str:
    audio_b64 = base64.b64encode(audio_bytes).decode()
    return (
        PLAYER_HTML.replace("__AUDIO__", audio_b64)
        .replace("__PRIMARIO__", PRIMARIO)
        .replace("__OSCURO__", OSCURO)
        .replace("__SUAVE__", SUAVE)
    )


# =====================================================================
#  LÓGICA: OCR
# =====================================================================
BASE = Path(__file__).parent


@st.cache_resource
def estado_tesseract():
    """Comprueba que Tesseract esté instalado y qué idiomas tiene."""
    try:
        pytesseract.get_tesseract_version()
        return True, set(pytesseract.get_languages(config=""))
    except Exception:
        return False, set()


def idioma_ocr(nombre: str, instalados: set):
    """Devuelve (idioma para Tesseract, aviso). Si falta el paquete, usa inglés."""
    datos = IDIOMAS[nombre]
    pack = datos["ocr"]
    if pack == "eng":
        return "eng", None
    if pack in instalados:
        return (f"{pack}+eng" if "eng" in instalados else pack), None
    return "eng", (
        f"Falta el paquete de OCR para {nombre}, así que se leerá como inglés. "
        f"Agrega `{datos['paquete']}` a `packages.txt` y reinicia la app."
    )


def aplicar_filtro(img_bgr, filtro: str):
    if filtro == "Sin filtro":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if filtro == "Escala de grises":
        return gris
    if filtro == "Invertir colores":
        return cv2.bitwise_not(gris)

    # Alto contraste: ampliar si es pequeña, suavizar y binarizar por zonas
    if gris.shape[1] < 1000:
        gris = cv2.resize(gris, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    return cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )


@st.cache_data(show_spinner=False, max_entries=20)
def reconocer(datos: bytes, filtro: str, idioma: str):
    """Decodifica la imagen, aplica el filtro y extrae el texto con Tesseract."""
    img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo leer la imagen.")
    procesada = aplicar_filtro(img, filtro)
    texto = pytesseract.image_to_string(procesada, lang=idioma).strip()
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return texto, original, procesada


# =====================================================================
#  LÓGICA: TRADUCCIÓN Y AUDIO
# =====================================================================
def unir_lineas(texto: str) -> str:
    """Convierte los saltos de línea del OCR en espacios (conserva los párrafos)."""
    texto = re.sub(r"-\n(?=\w)", "", texto)
    texto = re.sub(r"(?<!\n)\n(?!\n)", " ", texto)
    return re.sub(r"[ \t]{2,}", " ", texto).strip()


def dividir(texto: str, limite: int):
    """Parte el texto en trozos que respeten el límite de cada servicio."""
    partes, actual = [], ""
    for linea in texto.splitlines():
        while len(linea) > limite:
            if actual:
                partes.append(actual)
                actual = ""
            partes.append(linea[:limite])
            linea = linea[limite:]
        if len(actual) + len(linea) + 1 > limite and actual:
            partes.append(actual)
            actual = ""
        actual += linea + "\n"
    if actual.strip():
        partes.append(actual)
    return partes


@st.cache_data(show_spinner=False, ttl=3600)
def traducir_texto(texto: str, origen: str, destino: str) -> tuple[str, str]:
    """Traduce con Google (2 intentos) y, si está saturado, con MyMemory como respaldo.

    Devuelve (traducción, nombre del servicio usado).
    """
    if origen == destino:
        return texto, "Ninguno"

    ultimo_error = None
    for intento in range(2):
        try:
            traductor = GoogleTranslator(
                source=IDIOMAS[origen]["codigo"], target=IDIOMAS[destino]["codigo"]
            )
            partes = [traductor.translate(p) or "" for p in dividir(texto, 4500)]
            return "\n".join(partes).strip(), "Google"
        except Exception as error:
            ultimo_error = error
            time.sleep(1.5)

    try:
        traductor = MyMemoryTranslator(
            source=IDIOMAS[origen]["respaldo"], target=IDIOMAS[destino]["respaldo"]
        )
        partes = [traductor.translate(p) or "" for p in dividir(texto, 450)]
        return "\n".join(partes).strip(), "MyMemory"
    except Exception:
        raise ultimo_error


@st.cache_data(show_spinner=False, ttl=3600)
def generar_audio(texto: str, idioma: str, tld: str) -> bytes:
    """Genera el MP3 (con 2 intentos). Si ya se generó igual, usa la copia guardada."""
    ultimo_error = None
    for intento in range(2):
        try:
            buffer = io.BytesIO()
            gTTS(texto, lang=idioma, tld=tld, slow=False).write_to_fp(buffer)
            return buffer.getvalue()
        except Exception as error:
            ultimo_error = error
            time.sleep(1.5)
    raise ultimo_error


# =====================================================================
#  PÁGINA
# =====================================================================
with st.sidebar:
    st.subheader("Cómo usarlo")
    st.markdown(
        "1. Elige el idioma del texto de la imagen y el idioma al que quieres traducir.\n"
        "2. Toma una foto o sube una imagen.\n"
        "3. Revisa el texto reconocido y pulsa **Traducir y escuchar**."
    )
    st.subheader("Para mejores resultados")
    st.markdown(
        "- Buena luz, sin reflejos ni sombras.\n"
        "- Texto recto, ocupando buena parte de la foto.\n"
        "- Letra impresa: la manuscrita se reconoce mal."
    )

st.markdown(
    """
    <div class="hero">
        <div class="etiqueta">OCR · Traductor · Voz</div>
        <h1>De la imagen al audio</h1>
        <p>Fotografía un texto, tradúcelo a otro idioma y escúchalo en voz alta.</p>
        <div class="linea"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

ruta_imagen = BASE / IMAGEN
if ruta_imagen.is_file():
    st.image(Image.open(ruta_imagen), use_container_width=True)
    st.write("")

tesseract_ok, idiomas_instalados = estado_tesseract()

if not tesseract_ok:
    st.error(
        "Tesseract (el motor de reconocimiento) no está instalado en el servidor. "
        "Crea un archivo llamado `packages.txt` en la raíz del repositorio con este "
        "contenido y reinicia la app:"
    )
    st.code(
        "tesseract-ocr\ntesseract-ocr-spa\ntesseract-ocr-ben\n"
        "tesseract-ocr-kor\ntesseract-ocr-chi-sim\ntesseract-ocr-jpn",
        language="text",
    )
    st.stop()

# ---- 1. Idiomas y opciones ----
with st.container(border=True):
    st.subheader("1 · Idiomas")
    c1, c2 = st.columns(2)
    with c1:
        entrada = st.selectbox("Idioma del texto en la imagen", list(IDIOMAS), index=1)
    with c2:
        salida = st.selectbox("Traducir a", list(IDIOMAS), index=0)
    with st.expander("Opciones de imagen y voz"):
        filtro = st.selectbox(
            "Filtro de imagen",
            list(FILTROS),
            help="\n\n".join(f"**{n}:** {d}" for n, d in FILTROS.items()),
        )
        acento = st.selectbox("Acento de la voz", list(ACENTOS))

st.write("")

# ---- 2. Imagen ----
with st.container(border=True):
    st.subheader("2 · Imagen")
    origen = st.radio(
        "Origen de la imagen",
        ["📷 Cámara", "🖼️ Subir imagen"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if origen.startswith("📷"):
        st.caption(
            "Permite el acceso a la cámara cuando el navegador lo pida y centra "
            "el texto en el encuadre."
        )
        archivo = st.camera_input("Toma una foto", label_visibility="collapsed")
    else:
        archivo = st.file_uploader(
            "Sube una imagen",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )

# ---- 3. Texto reconocido ----
traducir = False
editado = ""
unir = True

if archivo is not None:
    st.write("")
    lang_ocr, aviso_ocr = idioma_ocr(entrada, idiomas_instalados)
    if aviso_ocr:
        st.warning(aviso_ocr)

    try:
        with st.spinner("Leyendo el texto..."):
            texto, original, procesada = reconocer(archivo.getvalue(), filtro, lang_ocr)
    except Exception as error:
        st.error(f"No se pudo analizar la imagen. ({error})")
        st.stop()

    with st.container(border=True):
        st.subheader("3 · Texto reconocido")
        i1, i2 = st.columns(2)
        with i1:
            st.image(original, caption="Imagen original", use_container_width=True)
        with i2:
            st.image(procesada, caption="Lo que lee el OCR", use_container_width=True)

        if texto:
            editado = st.text_area(
                "Texto reconocido",
                value=texto,
                height=220,
                label_visibility="collapsed",
            )
            unir = st.toggle(
                "Unir saltos de línea (mejor para párrafos)",
                value=True,
                help="El OCR separa el texto por renglones de la foto. Unirlos mejora la "
                "traducción y la lectura en voz alta.",
            )
            traducir = st.button(
                "Traducir y escuchar", type="primary", use_container_width=True
            )
        else:
            st.warning(
                "No se detectó texto. Prueba con más luz, acércate al texto, revisa el "
                "idioma elegido o cambia el filtro (por ejemplo, «Alto contraste»)."
            )

# ---- 4. Traducción y audio ----
if traducir:
    texto_final = unir_lineas(editado) if unir else editado.strip()
    if not texto_final:
        st.warning("El texto está vacío. Escribe o corrige algo antes de traducir.")
    else:
        traduccion = None
        servicio = ""
        with st.spinner("Traduciendo..."):
            try:
                traduccion, servicio = traducir_texto(texto_final, entrada, salida)
            except Exception:
                st.error(
                    "Los servicios de traducción están saturados en este momento. "
                    "Espera unos segundos y vuelve a pulsar el botón."
                )

        if traduccion:
            audio_traducido = None
            with st.spinner("Generando audio..."):
                try:
                    audio_traducido = generar_audio(
                        traduccion, IDIOMAS[salida]["codigo"], ACENTOS[acento]
                    )
                except Exception:
                    st.warning(
                        "La traducción está lista, pero no se pudo generar el audio "
                        "(el servicio de voz está saturado). Vuelve a intentarlo en unos segundos."
                    )
            st.session_state["resultado"] = {
                "texto": traduccion,
                "audio": audio_traducido,
                "idioma": salida,
                "servicio": servicio,
            }

resultado = st.session_state.get("resultado")
if resultado:
    st.write("")
    with st.container(border=True):
        st.subheader(f"Traducción · {resultado['idioma']}")
        st.markdown(
            f'<div class="resultado">{escape(resultado["texto"])}</div>',
            unsafe_allow_html=True,
        )
        if resultado["audio"]:
            components.html(reproductor(resultado["audio"]), height=120)
            st.download_button(
                "Descargar MP3",
                data=resultado["audio"],
                file_name="traduccion.mp3",
                mime="audio/mpeg",
                use_container_width=True,
            )
        if resultado.get("servicio") == "MyMemory":
            st.caption("Traducido con el servicio de respaldo (MyMemory).")

st.write("")
st.caption("OCR con Tesseract. Traducción y voz con servicios de Google (deep-translator y gTTS).")
