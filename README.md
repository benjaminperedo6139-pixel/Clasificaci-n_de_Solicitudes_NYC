# Clasificación de Solicitudes NYC 311: ¿se cierra en menos de 24 h?

> Informe final de la actividad. Cada sección indica su responsable; se completa en la rama de cada integrante.

## 1. Objetivo
<!-- Jose Carlos Isai Huitron Barron-->
Determinar qué modelo predice mejor si una solicitud de NYC 311 se cerrará en menos de 24 horas, haciendo la predicción en el momento en que se registra la solicitud.

## 2. Fuente de datos
<!-- Luis Angel Pardave Garcia -->
- NYC Open Data, *311 Service Requests from 2010 to Present* (API Socrata `erm2-nwe9`).
- Periodo: 2025-01-01 a 2025-03-31, muestra determinística del 10 % (`unique_key` termina en 0).
- Descarga: **88 503** solicitudes (≈ 983 por día). Tras los filtros del target quedan **86 656**.
- La primera ejecución de `python train.py` descarga los datos y los guarda en caché en `data/nyc311_ene_mar2025.json`.

## 3. Definición del target `P_24h`
<!-- Luis Angel Pardave Garcia -->

$$
P_{24h} = \begin{cases} 1 & \text{si } closed\_date - created\_date < 24\text{ h} \\ 0 & \text{si tardó } \ge 24\text{ h o sigue abierta tras } 24\text{ h} \end{cases}
$$

Implementado en [src/target.py](src/target.py).

| Caso | Solicitudes | Tratamiento |
|---|---|---|
| Cerradas en < 24 h | 49 540 | `P_24h = 1` |
| Cerradas en ≥ 24 h | 36 073 | `P_24h = 0` |
| **Siguen abiertas** (`closed_date` nulo) | 1 043 | `P_24h = 0`: llevan más de 24 h abiertas al momento de la descarga, así que su etiqueta ya se conoce. La versión de clase las descartaba sin querer (la diferencia de fechas era `NaT`), lo que sesgaba el target hacia la clase 1. |
| Duración negativa | 35 | Se descartan (error de captura). |
| Duración entre 0 y 150 s | 1 812 | Se descartan: son cierres automáticos o duplicados, no atención real (mismo filtro de clase). |

Proporción final: **57.2 % de solicitudes con `P_24h = 1`**, clases razonablemente balanceadas.

**¿Cuándo se conoce la etiqueta?** Como muy tarde en `created_date + 24 h`: para ese momento la solicitud ya se cerró (→ 1) o no se cerró a tiempo (→ 0). Esa fecha se guarda en la columna `label_known_at` y se usa para verificar cada corte (ver sección 6). Una solicitud abierta creada menos de 24 h antes de la descarga tendría etiqueta desconocida y se descartaría; en esta muestra no hay ninguna.

## 4. Variables disponibles al registrar la solicitud
<!-- Luis Angel Pardave Garcia -->

Solo se usan variables que existen **en el momento en que se registra la solicitud**:

| Variable | Tipo | Valores distintos | Nulos | Origen |
|---|---|---|---|---|
| `agency` | categórica | 14 | 0 | API (se asigna al registrar la solicitud) |
| `complaint_type` | categórica | 152 | 0 | API |
| `borough` | categórica | 6 | 0 | API |
| `open_data_channel_type` | categórica | 5 (ONLINE, PHONE, MOBILE, UNKNOWN, OTHER) | 0 | API |
| `location_type` | categórica | 87 | 9 252 (10.7 %) | API |
| `hour` | numérica | 0 a 23 | 0 | derivada de `created_date` |
| `weekday` | categórica | 7 | 0 | derivada de `created_date` |

**¿Por qué agregamos `agency`?** La dependencia responsable es la variable más asociada al target:

| Agencia | Solicitudes | % P_24h |
|---|---|---|
| NYPD (policía) | 38 280 | 97.3 % |
| HPD (vivienda) | 24 701 | 16.2 % |
| DSNY (limpieza) | 7 295 | 39.1 % |
| DOT (transporte) | 5 123 | 41.5 % |
| DEP (ambiente) | 4 015 | 47.9 % |
| Otras 9 | 7 242 | 0 % a 33 % |

Cada `complaint_type` pertenece como máximo a 2 agencias, así que `agency` es en parte redundante con `complaint_type`. Aun así ayuda a los modelos a agrupar tipos de queja poco frecuentes de la misma dependencia, y es la variable más fácil de interpretar.

**Se excluyen** `closed_date`, `status`, `resolution_description` y `duration_hours`, porque solo se conocen después del cierre: usarlas sería fuga de información. `closed_date` se usa únicamente para construir el target.

## 5. Dependencias y reproducción
```bash
pip install -r requirements.txt
python train.py            # descarga los datos (caché en data/) y corre el experimento
streamlit run app.py       # interfaz gráfica
```
Opcional: definir `SOCRATA_APP_TOKEN` para evitar límites de la API.

## 6. Validación temporal
<!-- Luis Angel Pardave Garcia -->

Implementado en [src/temporal_split.py](src/temporal_split.py).

**Diseño**
1. Las solicitudes se ordenan por `created_date`.
2. Los **últimos 14 días** (18 al 31 de marzo) se reservan como **prueba final**. No se usan para elegir el modelo ni sus hiperparámetros.
3. Con los días anteriores se construyen **3 cortes** con bloques de validación consecutivos de **14 días**.
4. **Ventana expansiva:** cada corte entrena con todo el pasado disponible. Así se simula la operación real (se reentrena con toda la historia acumulada) y se aprovechan más datos, algo importante para categorías poco frecuentes de `complaint_type` y `location_type`. Elegimos expansiva y no deslizante porque solo hay 3 meses de datos: recortar el entrenamiento dejaría el primer corte muy pequeño.
5. **Gap de 24 h:** una solicitud solo entra al entrenamiento si `created_date + 24 h ≤ inicio de la validación`. La etiqueta `P_24h` tarda hasta 24 h en conocerse, así que las solicitudes del último día antes del corte todavía no tenían etiqueta en la fecha del corte.

**¿Por qué cortes por días y no `TimeSeriesSplit` directamente?** `TimeSeriesSplit` corta por número de filas y su `gap` también se mide en filas. Como el volumen diario varía (de ≈ 720 a 1 500 solicitudes), cortar por fecha garantiza bloques de calendario completos y un gap de exactamente 24 h. La salida tiene el mismo formato que `TimeSeriesSplit.split()` (lista de `(train_idx, valid_idx)`), así que puede pasarse como `cv=` a `GridSearchCV`.

**Verificación de que las etiquetas ya se conocían.** En cada corte, y en el reentrenamiento final, se comprueba con `assert` que:
- toda solicitud de entrenamiento se creó antes del corte, y
- su `label_known_at` (`created_date + 24 h`) es menor o igual a la fecha del corte.

Además se probó que una fuga forzada (meter filas de validación al entrenamiento) hace fallar el `assert`.

| Corte | Train inicio | Train fin | Validación inicio | Validación fin | n train | n validación | % P_24h train | % P_24h validación |
|---|---|---|---|---|---|---|---|---|
| 1 | 2025-01-01 | 2025-02-03 00:00 | 2025-02-04 | 2025-02-17 | 35 789 | 12 074 | 56.2 % | 54.0 % |
| 2 | 2025-01-01 | 2025-02-16 | 2025-02-18 | 2025-03-03 | 47 878 | 12 961 | 55.8 % | 56.5 % |
| 3 | 2025-01-01 | 2025-03-03 00:00 | 2025-03-04 | 2025-03-17 | 60 775 | 12 560 | 55.9 % | 60.2 % |
| **Prueba final** | 2025-01-01 | 2025-03-16 | **2025-03-18** | **2025-03-31** | 73 287 | 12 356 | 56.6 % | 60.8 % |

La fila *Prueba final* muestra con qué datos se reentrena el modelo elegido (todo lo anterior al 17 de marzo, por el gap) y el bloque reservado donde se evalúa una sola vez. Los "Train fin" a las 00:00 aparecen porque la última solicitud con etiqueta conocida se creó exactamente 24 h antes del corte.

La proporción de `P_24h` sube de 54 % a 61 % hacia marzo (deriva temporal), y por eso es importante validar respetando el orden del tiempo en lugar de con un split aleatorio.

## 7. Métrica principal
<!-- Jose Carlos Isai Huitron Barron -->
**F1** de la clase positiva (`P_24h = 1`), definida antes de comparar resultados. Justificación: …

## 8. Modelos e hiperparámetros
<!-- Jorge Alberto Laureano Corona -->

| Modelo | Ventaja esperada | Limitación | Rejilla de hiperparámetros |
|---|---|---|---|
| Baseline (Dummy) | Fija el piso sin usar ninguna variable: un modelo que no lo supere claramente no está aprovechando los datos. Casi no cuesta entrenarlo. | No distingue entre solicitudes. Aun así su F1 no es bajo: prediciendo siempre 1, F1 = 2p/(1+p) ≈ 0.70 a 0.75 según la proporción p de cada validación (sección 6). | `strategy`: `most_frequent`, `stratified`<br>**2** combinaciones<br>Fijo: `random_state=42` |
| Regresión logística | Aprende un peso por columna one-hot, así que aprovecha directamente la asociación de `agency` y `complaint_type` con el target. La regularización (`C`) frena los pesos de categorías poco frecuentes. Rápida e interpretable. | Los efectos solo se suman: no capta interacciones (p. ej. un mismo `complaint_type` con distinto comportamiento según `borough` u `hour`). `hour` tiene un solo coeficiente: su efecto solo puede subir o bajar de 0 a 23, sin picos intermedios. | `C`: 0.01, 0.1, 1, 10<br>**4** combinaciones<br>Fijos: `max_iter=1000`, `random_state=42` |
| Random Forest | Puede capturar interacciones (p. ej. `complaint_type` × `borough` × `hour`) sin definirlas a mano y usar `hour` de forma no lineal con varios cortes. | Con one-hot cada división separa una sola categoría, así que aprovechar muchos de los 152 `complaint_type` exige árboles profundos; si se sortean pocas columnas por división, las informativas entran pocas veces. Puede sobreajustar categorías raras; es más costoso y menos interpretable. | `min_samples_leaf`: 1, 5, 20<br>`max_features`: `"sqrt"`, 0.2<br>**6** combinaciones<br>Fijos: `n_jobs=-1`, `random_state=42` |
| KNN | No supone una forma para la relación: predice con las solicitudes pasadas más parecidas, por lo que puede reflejar combinaciones locales (misma queja, mismo `borough`, hora cercana). | Cambiar de categoría suma la misma distancia en cualquier variable (`weekday` pesa igual que `complaint_type`); una diferencia grande en `hour` estandarizada puede pesar más, y las 23 h quedan lejos de las 0 h. Predecir es lento: compara con todas las filas de entrenamiento. | `n_neighbors`: 5, 15, 51, 101<br>`weights`: `uniform`, `distance`<br>**8** combinaciones<br>Fijos: ninguno |
| SVM | Un margen lineal regularizado suele funcionar bien con muchas columnas one-hot, y `LinearSVC` escala a ~73 000 filas. | Igual que la regresión logística, no capta interacciones y trata `hour` linealmente, así que se esperan resultados parecidos. No tiene `predict_proba`: su `decision_function` sirve para ROC AUC, pero no es una probabilidad. | `LinearSVC`<br>`C`: 0.001, 0.01, 0.1, 1<br>**4** combinaciones<br>Fijo: `random_state=42` |

Son expectativas previas, a contrastar con las secciones 9 y 10.

- **¿Por qué SVM lineal y no con kernel?** El tiempo de entrenamiento de `SVC` con kernel crece al menos con el cuadrado del número de filas: con ~73 000 filas y varias configuraciones por corte no es práctico, y usar una submuestra lo compararía con menos datos que a los demás. A cambio se renuncia a fronteras no lineales.
- **Limitación común:** ningún modelo corrige por sí solo la deriva temporal (sección 6), porque todos aprenden del pasado. Una categoría no vista en el entrenamiento queda en ceros en el one-hot (`handle_unknown="ignore"`).

**Búsqueda y selección.** Las rejillas ([src/models.py](src/models.py)) suman **24 combinaciones**; con 3 cortes son **72 ajustes** (24 × 3). Lo que no figura en la tabla queda en el valor por defecto de scikit-learn. Se elegirá el modelo y la configuración con mayor **F1 promedio en los 3 cortes**, sin usar el bloque de prueba.

**Estado:** las rejillas todavía **no se han evaluado con datos reales** y no hay resultados: la API no era accesible desde el equipo usado y [src/evaluation.py](src/evaluation.py) aún no está implementado. Solo se comprobó con datos **sintéticos** (90 000 filas, 271 columnas) que los 72 ajustes corren sin errores ni avisos, en unos 9 minutos en secuencial.

## 9. Resultados por corte
<!-- Jose Carlos Isai Huitron Barron -->

## 10. Resultado final en el bloque de prueba
<!-- Jose Carlos Isai Huitron Barron: métricas + matriz de confusión + significado de FP / FN -->

## 11. Conclusión
<!-- Todos -->

## 12. Interfaz gráfica
<!-- Benjamin Peredo Alonso -->

## 13. Contribuciones

| Integrante | Rama | Contribución | Commits |
|---|---|---|---|
| | `main` | Fase 0: estructura base, contratos entre módulos | |
| | `feature/datos-validacion` | Datos, target y validación temporal | |
| | `feature/modelos` | Modelos e hiperparámetros | |
| | `feature/evaluacion-reporte` | Evaluación, selección y reporte | |
| | `feature/streamlit` | Interfaz gráfica | |
