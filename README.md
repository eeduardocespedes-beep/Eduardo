
[README.md](https://github.com/user-attachments/files/26041948/README.md)
# 🏦 Predicción de Mora en Créditos Bancarios con Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-DNN-D00000?style=for-the-badge&logo=keras)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=for-the-badge&logo=googlecolab)

**Trabajo Final — Modelos de Deep Learning: Aplicaciones Prácticas**
**Autor:** Céspedes Alvarado, Edwin Eduardo | **Año:** 2026

</div>

---

## 📌 Descripción del Proyecto

Este proyecto aplica técnicas de **Deep Learning** para predecir la **mora crediticia** — el incumplimiento en el pago de cuotas de préstamos bancarios. Se construye, entrena y evalúa una **Red Neuronal Profunda (DNN)** con TensorFlow/Keras, comparándola con modelos baseline (Regresión Logística, Random Forest).

> 💡 **Relevancia real:** La tasa promedio global de NPL (Non-Performing Loans) es **5.8%** (Banco Mundial, 2022). En América Latina alcanzó **4.1%** en 2023, y en Perú se estima en **4.5%** (SBS, 2024). Para una cartera de S/ 800M con LGD=55%, esto representa más de **S/ 19.8M en pérdidas anuales** que un modelo predictivo puede reducir significativamente.

---

## 🎯 Objetivos

- Construir un pipeline completo de ML/DL para clasificación binaria de riesgo crediticio
- Comparar modelos clásicos vs. Deep Learning en términos de Accuracy, F1-Score y AUC-ROC
- Aplicar técnicas de balanceo de clases (SMOTE + class weighting)
- Interpretar el modelo con análisis **SHAP** para cumplimiento regulatorio
- Cuantificar el impacto económico del modelo en una institución financiera real

---

## 📊 Resultados Obtenidos

| Modelo | Accuracy | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| Regresión Logística | ~78% | ~0.71 | ~0.82 |
| Random Forest | ~86% | ~0.83 | ~0.91 |
| **DNN (Propuesto) ★** | **~93%** | **~0.91** | **~0.96** |

### Métricas del Modelo DNN (conjunto de prueba)

```
              precision    recall  f1-score   support
 Sin Mora (0)    0.94       0.96      0.95      5,623
 Con Mora (1)    0.89       0.85      0.87      2,377
 Macro Avg       0.91       0.90      0.91      8,000
```

**AUC-ROC: 0.962 | AUC-PR: 0.891 | Umbral de decisión: 0.40**

---

## 🏗️ Arquitectura del Modelo DNN

```
Input (20 features)
       ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.3)
       ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.3)
       ↓
Dense(64)  → ReLU → Dropout(0.2)
       ↓
Dense(32)  → ReLU → Dropout(0.2)
       ↓
Dense(1)   → Sigmoid → Probabilidad de mora [0, 1]
```

- **Optimizador:** Adam (lr=0.001)
- **Loss:** Binary Crossentropy + class_weight {0:1, 1:3}
- **Regularización:** BatchNormalization + Dropout + L2
- **Callbacks:** EarlyStopping (patience=15) + ReduceLROnPlateau

---

## 📁 Estructura del Repositorio

```
📦 DL-CreditDefault-MoraPrediction/
│
├── 📓 DL-Final-CespedesAlvarado-EdwinEduardo.ipynb   ← Notebook principal
│
├── 📊 data/
│   └── credit_default_data.csv     ← Dataset sintético (30,000 registros)
│
├── 📋 requirements.txt             ← Dependencias del proyecto
│
├── 📝 informe/
│   └── Informe_Mora_DeepLearning.docx  ← Informe académico completo
│
└── 📖 README.md                    ← Este archivo
```

---

## 📦 Dataset

El dataset utilizado es una versión sintética representativa del **UCI Credit Default Dataset** (Default of Credit Card Clients, Taiwan 2005), con las siguientes características:

| Característica | Valor |
|---|---|
| Total de registros | 30,000 |
| Variables predictoras | 20 |
| Variable objetivo | `DEFAULT` (0=Sin mora, 1=Con mora) |
| Tasa de mora | ~22% |
| Fuente de referencia | UCI ML Repository |

### Variables principales

| Variable | Descripción | Importancia |
|---|---|---|
| `payment_ratio` | Cuotas pagadas / Total cuotas | ★★★★★ |
| `days_overdue` | Días de atraso acumulados | ★★★★★ |
| `credit_score` | Score crediticio (300–850) | ★★★★☆ |
| `debt_to_income` | Deuda total / Ingresos | ★★★★☆ |
| `num_prev_defaults` | Moras previas históricas | ★★★★☆ |

---

## 🚀 Cómo Ejecutar

### Opción 1: Google Colab (recomendado)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eeduardocespedes-beep/DL-Final-CespedesAlvarado-EdwinEduardo/blob/main/DL-Final-CespedesAlvarado-EdwinEduardo.ipynb)

### Opción 2: Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/eeduardocespedes-beep/DL-Final-CespedesAlvarado-EdwinEduardo.git
cd DL-Final-CespedesAlvarado-EdwinEduardo

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Lanzar Jupyter
jupyter notebook DL-Final-CespedesAlvarado-EdwinEduardo.ipynb
```

---

## 🔧 Pipeline del Notebook

```
1. 📦 Instalación de dependencias
2. 📊 Generación/carga del dataset + EDA (8 visualizaciones)
3. 🔧 Preprocesamiento: StandardScaler + SMOTE
4. 📈 Modelos Baseline: Regresión Logística + Random Forest
5. 🧠 Modelo DNN con TensorFlow/Keras (4 capas ocultas)
6. 📉 Evaluación: Matriz de confusión, ROC, Precision-Recall
7. 🔍 Interpretabilidad SHAP (importancia de variables)
8. 💰 Simulador de impacto económico parametrizable
9. 🎯 Predicción individual para nuevos clientes
10. 📋 Resumen ejecutivo consolidado
```

---

## 💰 Impacto Económico

Para una institución financiera con cartera de **S/ 800 millones** (referencia SBS Perú 2024):

| Métrica | Valor |
|---|---|
| Cartera morosa actual (4.5%) | S/ 36.0 M |
| Pérdida esperada sin modelo (LGD 55%) | S/ 19.8 M/año |
| Clientes morosos detectados (recall 85%) | 2,020 de 2,377 |
| Pérdida evitada con el modelo | **S/ 6.0 M/año** |
| ROI estimado del proyecto | **>1,400%** |

---

## 🔍 Interpretabilidad SHAP

El modelo incluye análisis SHAP completo para cumplir con los requerimientos de **transparencia algorítmica** del Comité de Basilea (SR 11-7) y la SBS Perú:

- Importancia global de variables (bar plot + beeswarm)
- Distribución de impacto por variable (violin plot)
- Top 10 variables ordenadas por valor SHAP medio
