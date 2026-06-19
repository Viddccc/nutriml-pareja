# NutriML-Pareja 🤰

**Sistema predictivo con Machine Learning para evaluación nutricional preconcepcional en parejas**  
Universidad Nacional de Juliaca (UNAJ) — Ingeniería de Software y Sistemas — 2026  
Autor: Condori Ccosi Vidal

---

## Instalación y ejecución local

### 1. Requisitos previos
- Python 3.10 o superior
- pip actualizado

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación
```bash
streamlit run app.py
```
Se abrirá automáticamente en tu navegador en `http://localhost:8501`

---

## Archivos necesarios (misma carpeta que app.py)
```
nutriml_app/
├── app.py              ← Aplicación Streamlit
├── modelos.pkl         ← Modelos ML entrenados (5 algoritmos)
├── preprocesador.pkl   ← Pipeline de preprocesamiento
├── requirements.txt    ← Dependencias Python
└── README.md           ← Este archivo
```

---

## Deploy gratuito en Streamlit Cloud

1. Sube esta carpeta a un repositorio GitHub
2. Ve a https://share.streamlit.io
3. Conecta tu cuenta GitHub
4. Selecciona el repositorio y `app.py` como archivo principal
5. Haz clic en "Deploy" — en 2 minutos tendrás una URL pública

---

## Variables del sistema

### Variables maternas (8)
| Variable | Unidad | Ref. normal (Juliaca) |
|----------|--------|-----------------------|
| IMC materno | kg/m² | 18.5–24.9 |
| Hemoglobina materna ajustada* | g/dL | ≥ 14.0 |
| Ferritina sérica | μg/L | ≥ 15 |
| Ácido fólico eritrocitario | nmol/L | ≥ 305 |
| Zinc sérico | μg/dL | ≥ 70 |
| Ganancia de peso pregestacional | kg | 11.5–16.0 |
| Edad materna | años | 18–35 |
| Paridad | n° embarazos | — |

### Variables paternas (4)
| Variable | Unidad | Ref. normal (Juliaca) |
|----------|--------|-----------------------|
| IMC paterno | kg/m² | 18.5–24.9 |
| Hemoglobina paterna ajustada* | g/dL | ≥ 15.0 |
| Edad paterna | años | — |
| Diversidad dietética | 0–10 | ≥ 6 |

*Corrección OPS/OMS para altitud 3 824 m s.n.m.: valor medido − 2.0 g/dL

---

## Outcomes predichos
- **Bajo peso al nacer (BPN):** < 2 500 g
- **Peso adecuado:** 2 500–3 999 g
- **Macrosomía:** ≥ 4 000 g

---

## Algoritmos disponibles
1. Árbol de Decisión (baseline)
2. Regresión Logística
3. **SVM** ← mejor modelo (F1=0.736, ACC=76.5%)
4. Random Forest
5. XGBoost

---

## Nota importante
Los modelos (.pkl) fueron entrenados con datos sintéticos calibrados a las 
distribuciones de la región altiplánica de Puno. Para uso clínico real, 
reentrenar con el dataset del Hospital Carlos Monge Medrano de Juliaca.
