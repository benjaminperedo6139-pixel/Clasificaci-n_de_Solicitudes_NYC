# Clasificación de Solicitudes NYC 311: ¿se cierra en menos de 24 h?

> Informe final de la actividad. Cada sección indica su responsable; se completa en la rama de cada integrante.

## 1. Objetivo
<!-- Integrante C -->
Determinar qué modelo predice mejor si una solicitud de NYC 311 se cerrará en menos de 24 horas, haciendo la predicción en el momento en que se registra la solicitud.

## 2. Fuente de datos
<!-- Integrante A -->
- NYC Open Data, *311 Service Requests from 2010 to Present* (API Socrata `erm2-nwe9`).
- Periodo: 2025-01-01 a 2025-03-31, muestra determinística del 10 % (`unique_key` termina en 0).

## 3. Definición del target `P_24h`
<!-- Integrante A: definición, filtros, solicitudes abiertas y verificación de que la etiqueta se conocía en cada corte -->

## 4. Variables disponibles al registrar la solicitud
<!-- Integrante A -->

## 5. Dependencias y reproducción
```bash
pip install -r requirements.txt
python train.py            # descarga los datos (caché en data/) y corre el experimento
streamlit run app.py       # interfaz gráfica
```
Opcional: definir `SOCRATA_APP_TOKEN` para evitar límites de la API.

## 6. Validación temporal
<!-- Integrante A: justificación (ventana expansiva, gap de 24 h) + tabla -->

| Corte | Train inicio | Train fin | Validación inicio | Validación fin | n train | n validación |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| Prueba final | — | — | | | — | |

## 7. Métrica principal
<!-- Integrante C -->
**F1** de la clase positiva (`P_24h = 1`), definida antes de comparar resultados. Justificación: …

## 8. Modelos e hiperparámetros
<!-- Integrante B -->

| Modelo | Ventaja esperada | Limitación | Hiperparámetros probados |
|---|---|---|---|
| Baseline (Dummy) | | | |
| Regresión logística | | | |
| Random Forest | | | |
| KNN | | | |
| SVM | | | |

## 9. Resultados por corte
<!-- Integrante C -->

## 10. Resultado final en el bloque de prueba
<!-- Integrante C: métricas + matriz de confusión + significado de FP / FN -->

## 11. Conclusión
<!-- Todos -->

## 12. Interfaz gráfica
<!-- Integrante D -->

## 13. Contribuciones

| Integrante | Rama | Contribución | Commits |
|---|---|---|---|
| | `main` | Fase 0: estructura base, contratos entre módulos | |
| | `feature/datos-validacion` | Datos, target y validación temporal | |
| | `feature/modelos` | Modelos e hiperparámetros | |
| | `feature/evaluacion-reporte` | Evaluación, selección y reporte | |
| | `feature/streamlit` | Interfaz gráfica | |
