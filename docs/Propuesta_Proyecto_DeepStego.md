**Formulario de**
**Propuesta de Proyecto**
**TEL-328**
**Procesamiento Digital de Imágenes**

|  |  |  |
| --- | --- | --- |
| **Fecha** |  | 10 de septiembre de 2026 |
| **Nombre del proyecto** |  | DeepStego: mensajes ocultos en imágenes con redes neuronales |
| **Integrantes del equipo** |  | Abraham Garcia |
|  |  | Diego Acevedo |
|  |  | Juan Monteverde |
|  |  | Ignacio Figueroa |
|  |  | Yair Flores |

# **1. Introducción**

La esteganografía es la técnica de ocultar información dentro de un medio de apariencia inofensiva, de modo que un observador no advierta que existe un mensaje. A diferencia del cifrado, que protege el contenido pero deja en evidencia que hay algo que proteger, la esteganografía oculta la existencia misma de la comunicación. Este proyecto propone DeepStego, un prototipo que utiliza redes neuronales convolucionales para ocultar un mensaje de texto, codificado como código QR, dentro de una fotografía de apariencia normal, y recuperarlo después, adaptando el enfoque de esteganografía profunda de Baluja [1]. El uso de códigos QR entrega además un criterio objetivo de funcionamiento: si el código recuperado puede leerse con un lector estándar, la información oculta sobrevivió al proceso. El sistema se ofrecerá mediante una interfaz web sencilla.

1.1 Diagnóstico e identificación del problema, desafío u oportunidad

Las comunicaciones digitales están cada vez más expuestas a vigilancia y control. Según Freedom House, la libertad en internet disminuyó a nivel mundial por decimoquinto año consecutivo: de los 72 países evaluados en su informe 2025, las condiciones empeoraron en 28 y mejoraron solo en 17 [2]. En estos contextos, cifrar un mensaje no siempre es suficiente, porque un contenido cifrado es evidente: puede ser bloqueado o levantar sospechas sobre quien lo envía.

Las imágenes son un canal natural para comunicarse sin llamar la atención. Se estima que cada día se comparten cerca de 14 mil millones de imágenes en redes sociales, de las cuales alrededor de 6,9 mil millones corresponden solo a WhatsApp [3]. Una fotografía más dentro de ese volumen no despierta sospechas.

Sin embargo, los métodos clásicos de esteganografía en imágenes, como la sustitución de los bits menos significativos (LSB), tienen una limitación importante: o bien ocultan poca información, o bien, al aumentar la cantidad oculta, introducen distorsiones visibles y huellas estadísticas que las técnicas de esteganálisis pueden detectar [4], [5]. Los métodos basados en aprendizaje profundo abren una oportunidad, ya que la red aprende por sí misma dónde y cómo distribuir la información, lo que permite ocultar incluso imágenes completas con baja distorsión visible [1], [6]. Su principal desafío es que la información recuperada es una aproximación y no una copia exacta. Los códigos QR permiten abordar ese desafío, porque su corrección de errores permite leerlos aun con cerca de un 30 % de sus datos dañados [7].

1.2 Usuarios y clientes

El proyecto involucra a los siguientes actores:

**Usuarios que requieren comunicación discreta (usuarios directos).** Periodistas y sus fuentes, activistas, defensores de derechos humanos y, en general, personas que valoran su privacidad. Su interés es enviar información sin que un tercero sepa que existe un mensaje. Necesitan una herramienta simple, que no exija conocimientos técnicos y que no altere visiblemente las imágenes.

**Estudiantes y docentes (usuarios académicos).** Personas que estudian procesamiento de imágenes, aprendizaje profundo o seguridad informática. Su interés es contar con un ejemplo práctico y reproducible de redes codificador-decodificador aplicadas a un problema real. Necesitan código documentado y métricas claras.

**Profesionales de ciberseguridad y análisis forense (actores indirectos).** Les interesa comprender cómo se oculta información mediante redes neuronales para poder detectarla. Necesitan conocer las capacidades y los límites de estas técnicas.

**Observador del canal (adversario).** Censores, intermediarios o cualquier tercero que monitorea las comunicaciones. Su interés es detectar o bloquear mensajes; el sistema busca que solo vea imágenes comunes.

**Equipo docente de TEL-328 (cliente académico).** Evalúa el prototipo y requiere que cumpla los requisitos y plazos definidos en esta propuesta.

1.3 Propuesta de solución

Se propone desarrollar DeepStego, un prototipo de software que permite ocultar un mensaje de texto corto dentro de una fotografía y recuperarlo posteriormente. El mensaje se convierte primero en un código QR, que actúa como imagen secreta. El resultado del ocultamiento es una imagen contenedora que, a simple vista, no se distingue de la fotografía original (imagen portadora).

El núcleo de la solución son dos redes neuronales convolucionales entrenadas en conjunto, adaptando la arquitectura de Baluja [1]. La red de ocultamiento recibe la fotografía portadora y el código QR, y genera la imagen contenedora. La red de revelado recibe solo la imagen contenedora y reconstruye el código QR. Durante el entrenamiento se minimizan al mismo tiempo dos errores: la diferencia entre portadora y contenedora, para que el ocultamiento sea imperceptible, y la diferencia entre el QR original y el revelado, para que la recuperación sea fiel. Emisor y receptor deben compartir el modelo entrenado, que cumple un rol similar al de una clave.

Se eligió ocultar códigos QR en lugar de fotografías completas porque contienen mucha menos información: son imágenes en blanco y negro, sin colores ni texturas. Esto facilita el entrenamiento, permite alcanzar mejores resultados con recursos de cómputo gratuitos y entrega un criterio claro de éxito, ya que el QR se lee o no se lee.

Los QR se generarán con corrección de errores de nivel H. El QR revelado se limpiará con técnicas clásicas de procesamiento de imágenes, como suavizado y umbralización de Otsu [8], y luego se decodificará con un lector estándar. Se considerará que el prototipo funciona si, en fotografías no vistas durante el entrenamiento, se lee correctamente al menos el 90 % de los QR ocultos, manteniendo entre portadora y contenedora un PSNR de al menos 30 dB y un SSIM de al menos 0,90 [9].

El desarrollo se realizará en Python con PyTorch y OpenCV. El entrenamiento se hará en GPU gratuitas de Google Colab y Kaggle, con imágenes de 128×128 píxeles para mantener acotados los tiempos de entrenamiento. Los contenidos de la Unidad 2 del curso (redes convolucionales, autoencoders y redes generativas) entregan la base conceptual del proyecto. El prototipo se ofrecerá mediante una interfaz web construida con Gradio.

En cuanto a trabajos previos y alternativas, existen herramientas de esteganografía clásica, como Steghide u OpenStego, que ocultan archivos o texto modificando bits de la imagen. En investigación, Baluja [1] demostró que es posible ocultar una imagen a color completa dentro de otra usando redes neuronales; HiDDeN [10] y SteganoGAN [11] ocultan mensajes binarios; HiNet [6] mejora la calidad mediante redes invertibles, y StegaStamp [12] logra mensajes que sobreviven a fotografías impresas. Estos trabajos existen principalmente como artículos y código orientado a especialistas, no como herramientas de uso sencillo. DeepStego adapta la idea de Baluja al caso de códigos QR.

**Alcance.** El núcleo del proyecto es ocultar y recuperar códigos QR en imágenes guardadas en formato PNG, sin compresión con pérdida. Como extensión opcional, no comprometida, si el tiempo lo permite se probará ocultar fotografías completas como imagen secreta, lo que solo requiere cambiar los datos de entrenamiento. La robustez frente a compresión JPEG (por ejemplo, al enviar imágenes por redes sociales) y la detección de imágenes con contenido oculto quedan fuera del alcance y se plantean como trabajo futuro.

1.4 Atributos Diferenciadores

A diferencia de las herramientas clásicas, que modifican bits fijos de la imagen y dejan patrones detectables, DeepStego aprende a distribuir la información oculta en toda la fotografía, con baja distorsión visible. A diferencia de los trabajos de investigación en esteganografía profunda, publicados como código para especialistas, DeepStego integra el modelo en una interfaz web que cualquier persona puede usar: escribir un mensaje, ocultarlo en una foto y descargar el resultado.

Su principal atributo diferenciador es la verificación funcional mediante códigos QR. En lugar de juzgar la recuperación solo por su parecido visual, el sistema comprueba si el mensaje del QR puede leerse, lo que entrega un criterio objetivo de éxito o fracaso. Además, combina aprendizaje profundo con procesamiento clásico de imágenes: la umbralización de Otsu y la corrección de errores del QR compensan las imperfecciones propias de la reconstrucción neuronal, aprovechando las fortalezas de ambos enfoques.

1.5 Referencias

[1] S. Baluja, “Hiding images in plain sight: Deep steganography,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017.

[2] Freedom House, “Freedom on the Net 2025: An uncertain future for the global internet,” 2025. [Online]. Disponible: https://freedomhouse.org/report/freedom-net/2025/uncertain-future-global-internet

[3] M. Broz, “Photo statistics: How many photos are taken every day?,” Photutorial, 2026. [Online]. Disponible: https://photutorial.com/photos-statistics/

[4] J. Fridrich, Steganography in Digital Media: Principles, Algorithms, and Applications. Cambridge, U.K.: Cambridge University Press, 2009.

[5] A. Cheddad, J. Condell, K. Curran y P. Mc Kevitt, “Digital image steganography: Survey and analysis of current methods,” Signal Processing, vol. 90, no. 3, pp. 727–752, 2010.

[6] J. Jing, X. Deng, M. Xu, J. Wang y Z. Guan, “HiNet: Deep image hiding by invertible network,” in Proc. IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 4733–4742.

[7] ISO/IEC 18004:2015, Information technology — Automatic identification and data capture techniques — QR Code bar code symbology specification, 2015.

[8] N. Otsu, “A threshold selection method from gray-level histograms,” IEEE Transactions on Systems, Man, and Cybernetics, vol. 9, no. 1, pp. 62–66, 1979.

[9] Z. Wang, A. C. Bovik, H. R. Sheikh y E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.

[10] J. Zhu, R. Kaplan, J. Johnson y L. Fei-Fei, “HiDDeN: Hiding data with deep networks,” in Proc. European Conference on Computer Vision (ECCV), 2018, pp. 657–672.

[11] K. A. Zhang, A. Cuesta-Infante, L. Xu y K. Veeramachaneni, “SteganoGAN: High capacity image steganography with GANs,” arXiv:1901.03892, 2019.

[12] M. Tancik, B. Mildenhall y R. Ng, “StegaStamp: Invisible hyperlinks in physical photographs,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 2117–2126.

[13] E. Agustsson y R. Timofte, “NTIRE 2017 challenge on single image super-resolution: Dataset and study,” in Proc. IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2017.

[14] T.-Y. Lin et al., “Microsoft COCO: Common objects in context,” in Proc. European Conference on Computer Vision (ECCV), 2014, pp. 740–755.

# **2. Requisitos del Sistema**

Esta sección describe los requisitos funcionales de DeepStego y los recursos de hardware, software y datos que se utilizarán. El usuario esperado es una persona sin conocimientos de aprendizaje profundo, que interactúa con el sistema a través de una interfaz web: carga una fotografía y escribe un mensaje, y obtiene una imagen descargable con el mensaje oculto o el mensaje ya decodificado. Todos los recursos considerados son gratuitos, por lo que el proyecto no requiere inversión.

## 2.1 Requisitos Funcionales

Los requisitos funcionales del sistema son los siguientes:

* **RF1. Ocultar:** el sistema debe recibir una fotografía portadora y un código QR, y generar una imagen contenedora visualmente indistinguible de la portadora, con PSNR ≥ 30 dB y SSIM ≥ 0,90 respecto de ella.
* **RF2. Revelar:** a partir únicamente de la imagen contenedora, el sistema debe reconstruir una aproximación del código QR oculto.
* **RF3. Generar QR desde texto:** el sistema debe permitir escribir un mensaje corto (hasta 20 caracteres) y convertirlo automáticamente en el código QR que se ocultará.
* **RF4. Leer QR revelado:** el sistema debe limpiar y decodificar el QR revelado, mostrando el mensaje de texto. Debe leer correctamente al menos el 90 % de los QR del conjunto de prueba.
* **RF5. Evaluar calidad:** el sistema debe calcular PSNR y SSIM entre portadora y contenedora, y la tasa de lectura de QR sobre un conjunto de prueba.
* **RF6. Interfaz web:** el sistema debe ofrecer una interfaz web donde el usuario pueda cargar una fotografía, escribir un mensaje, ocultarlo, revelarlo y descargar la imagen contenedora en formato PNG.

### 2.2 Hardware

El proyecto no contempla la compra de hardware. Se utilizarán los siguientes recursos:

* **Computadores personales de los integrantes:** desarrollo de código, pruebas y ejecución de la interfaz web. Una vez entrenado, el modelo es liviano y puede ejecutarse en CPU.
* **GPU en la nube de uso gratuito (Google Colab y Kaggle Notebooks):** entrenamiento de las redes neuronales. Ambos servicios ofrecen GPU NVIDIA con cuotas de uso limitadas, por lo que los entrenamientos se distribuirán entre ambas plataformas y entre las cuentas personales de los integrantes.
* **Almacenamiento gratuito en la nube (Google Drive):** respaldo de conjuntos de datos, modelos entrenados y resultados.
* **Teléfono celular con cámara:** lectura de los códigos QR revelados durante la demostración en vivo, como verificación con un lector externo al sistema.

### 2.3 Software

Todo el software utilizado es gratuito:

* **Python 3:** lenguaje principal del proyecto.
* **PyTorch y torchvision:** definición, entrenamiento y ejecución de las redes neuronales.
* **OpenCV:** lectura y escritura de imágenes, suavizado, umbralización de Otsu, y detección y decodificación de códigos QR.
* **NumPy y scikit-image:** operaciones numéricas y cálculo de las métricas PSNR y SSIM.
* **qrcode (librería de Python):** generación de códigos QR.
* **Matplotlib:** gráficos del entrenamiento y visualización de resultados.
* **Gradio:** interfaz web del prototipo.
* **Jupyter Notebook, Google Colab y Kaggle:** entornos de desarrollo y entrenamiento.
* **Git y GitHub:** control de versiones, trabajo colaborativo y entrega del código fuente.

### 2.4 Datos de Imágenes/Videos

No se recolectarán datos personales. Se utilizarán tres fuentes de imágenes:

* **DIV2K [13]:** conjunto público de imágenes de alta resolución y contenido variado, de uso libre para fines académicos. Se usarán sus 900 imágenes públicas (800 de entrenamiento y 100 de validación), de las que se extraerán recortes aleatorios de 128×128 píxeles como imágenes portadoras.
* **COCO 2017 [14]:** se tomará un subconjunto de 1.000 imágenes de su partición de validación como portadoras de prueba. Al provenir de una fuente distinta a la de entrenamiento, permite evaluar si el modelo funciona con fotografías nuevas.
* **Códigos QR sintéticos:** se generarán con la librería qrcode a partir de textos aleatorios de hasta 20 caracteres, con corrección de errores de nivel H, en blanco y negro y a 128×128 píxeles. Serán las imágenes secretas. Para entrenamiento se generarán sobre la marcha, y para prueba se fijará un conjunto de 1.000 códigos con semilla fija.

**Composición del entrenamiento.** Cada ejemplo es un par (portadora, secreta), donde la portadora es una fotografía y la secreta es un código QR. En la extensión opcional se usarían también fotografías como imágenes secretas, sin necesidad de nuevas fuentes de datos.

# **3. Planificación del Proyecto**

La planificación se organiza en torno a los hitos de evaluación del curso: bitácora 1 (21/10), presentación de avance (22/10), entrega de bitácora 2, manual de uso y código fuente (18/11), demostración en vivo (19/11) y video demo (22/11). No se programan actividades durante las vacaciones de estudiantes (14 al 18/09).

### 3.1 Objetivo General

Desarrollar un prototipo funcional, basado en redes neuronales convolucionales, que permita ocultar un mensaje, codificado como código QR, dentro de una fotografía de forma visualmente imperceptible y recuperarlo, validando su funcionamiento mediante la lectura del QR revelado y ofreciéndolo a través de una interfaz web de uso sencillo.

### 3.2 Objetivos Específicos

Para cumplir el objetivo general, el proyecto debe:

* **O1. Construir el conjunto de datos:** preparar las fotografías portadoras (DIV2K y COCO) y generar los códigos QR sintéticos, separando datos de entrenamiento, validación y prueba.
* **O2. Implementar y entrenar el modelo de ocultamiento y revelado:** lograr PSNR ≥ 30 dB y SSIM ≥ 0,90 entre imagen portadora y contenedora en el conjunto de prueba.
* **O3. Implementar la lectura de QR revelados:** decodificar correctamente al menos el 90 % de los QR ocultos en imágenes de prueba.
* **O4. Evaluar el sistema:** medir PSNR, SSIM y tasa de lectura de QR, y documentar los resultados y limitaciones encontradas.
* **O5. Desarrollar la interfaz web:** integrar todos los módulos en una aplicación que permita a un usuario sin conocimientos técnicos ocultar, revelar y descargar imágenes.

La extensión opcional de ocultar fotografías completas no forma parte de los objetivos comprometidos.

### 3.3 Actividades

Cada actividad indica entre paréntesis el objetivo específico al que contribuye. En las columnas de responsable y participación se usan las iniciales de los integrantes: AG (Abraham Garcia), DA (Diego Acevedo), JM (Juan Monteverde), IF (Ignacio Figueroa) y YF (Yair Flores).

| **Actividad** | **Resp.** | **Participación equipo** | **Fecha de inicio** | **Fecha de término** | **% de avance** |
| --- | --- | --- | --- | --- | --- |
| **A1. Formulación de la propuesta:** alcance, requisitos, objetivos y planificación. | AG | Todos (20 % c/u) | 07/09/2026 | 11/09/2026 | 100 % |
| **A2. Revisión bibliográfica:** estudio de Baluja [1], HiNet [6] e implementaciones abiertas. (O2) | AG | AG 40 %, YF 30 %, JM 30 % | 21/09/2026 | 27/09/2026 | 0 % |
| **A3. Preparación del entorno:** repositorio GitHub, notebooks en Colab/Kaggle y estructura del código. (O1–O5) | JM | JM 60 %, DA 40 % | 21/09/2026 | 25/09/2026 | 0 % |
| **A4. Conjunto de datos (M1):** descarga de DIV2K y COCO, generador de QR y carga por lotes. (O1) | DA | DA 60 %, IF 40 % | 21/09/2026 | 04/10/2026 | 0 % |
| **A5. Redes de ocultamiento y revelado (M2):** implementación y entrenamiento de prueba. (O2) | YF | YF 60 %, DA 40 % | 28/09/2026 | 11/10/2026 | 0 % |
| **A6. Lectura de QR (M3):** suavizado, umbralización de Otsu y decodificación. (O3) | IF | IF 70 %, JM 30 % | 28/09/2026 | 11/10/2026 | 0 % |
| **A7. Módulo de evaluación (M4):** PSNR, SSIM y tasa de lectura de QR. (O4) | IF | IF 60 %, AG 40 % | 05/10/2026 | 18/10/2026 | 0 % |
| **A8. Interfaz web, primera versión (M5):** ocultar y revelar desde el navegador. (O5) | JM | JM 70 %, AG 30 % | 05/10/2026 | 20/10/2026 | 0 % |
| **A9. Entrenamiento principal:** entrenamiento completo del modelo y primera evaluación. (O2, O4) | YF | YF 60 %, DA 40 % | 12/10/2026 | 20/10/2026 | 0 % |
| **A10. Bitácora 1 y presentación de avance.** (O1–O4) | AG | Todos (20 % c/u) | 15/10/2026 | 22/10/2026 | 0 % |
| **A11. Ajuste del modelo:** hiperparámetros y largo de los mensajes según los resultados del avance. (O2, O3) | YF | YF 50 %, DA 30 %, IF 20 % | 23/10/2026 | 08/11/2026 | 0 % |
| **A12. Integración y pruebas:** todos los módulos en la interfaz y pruebas de extremo a extremo. (O5) | DA | DA 60 %, JM 40 % | 26/10/2026 | 13/11/2026 | 0 % |
| **A13. Evaluación final:** métricas en el conjunto de prueba y análisis de casos de falla. (O4) | IF | IF 60 %, AG 40 % | 02/11/2026 | 13/11/2026 | 0 % |
| **A14. Documentación:** bitácora 2, manual de uso y código fuente comentado. (O4, O5) | AG | AG 40 %, resto 15 % c/u | 09/11/2026 | 18/11/2026 | 0 % |
| **A15. Demostración en vivo:** preparación y presentación del prototipo. (O5) | AG | Todos (20 % c/u) | 16/11/2026 | 19/11/2026 | 0 % |
| **A16. Video demo:** grabación y edición del video del proyecto. (O5) | JM | JM 40 %, resto 15 % c/u | 19/11/2026 | 22/11/2026 | 0 % |

# **4. Diseño de la Solución**

### 4.1 Diagrama de Contexto

La Figura 1 resume el funcionamiento de DeepStego. El emisor escribe un mensaje, que se convierte en código QR y se oculta dentro de una fotografía portadora. La imagen contenedora resultante se ve igual a la original y se envía por un canal sin compresión. El receptor, con el mismo modelo entrenado, revela el QR oculto y el sistema lee el mensaje. Todo el proceso se realiza desde la interfaz web.

![](data:image/png;base64...)
Figura 1: Diagrama de contexto del sistema

4.2 Módulos

El sistema se compone de cinco módulos. M1 prepara los datos, M2 contiene las redes neuronales de ocultamiento y revelado, M3 lee los códigos QR revelados, M4 evalúa el desempeño y M5 integra todo en la interfaz web.

**M1. Módulo de Datos (DAT)**

|  |  |
| --- | --- |
| **Propósito** | Preparar y entregar los datos para entrenar y evaluar el sistema: fotografías portadoras y códigos QR sintéticos. En la interfaz, también convierte el mensaje del usuario en un QR y ajusta la fotografía cargada al tamaño de trabajo. |
| **Marco Técnico** | Recorte aleatorio y redimensionamiento de fotografías a 128×128 píxeles, normalización de intensidades al rango [0, 1] y aumento de datos (volteo horizontal). Generación de QR con corrección de errores de nivel H según ISO/IEC 18004 [7], en escala de grises (1 canal) y escalados con interpolación por vecino más cercano para conservar bordes nítidos. Separación fija de entrenamiento, validación y prueba mediante semilla aleatoria. |
| **Estructura General** | **Entradas:** fotografías de DIV2K y COCO; textos aleatorios o mensaje escrito por el usuario.  **Ejecución:** (1) cargar y recortar fotografías; (2) generar un QR a partir de un texto aleatorio; (3) formar pares (portadora, QR); (4) agrupar en lotes.  **Salidas:** lotes de portadoras (lote, 3, 128, 128) y de QR (lote, 1, 128, 128) para M2, y un conjunto de prueba fijo para M4. |
| **Requisito Funcional** | RF3 (y datos de soporte para RF1, RF2 y RF5). |

**M2. Módulo de Esteganografía Neuronal (EST)**

|  |  |
| --- | --- |
| **Propósito** | Núcleo del sistema. Contiene la red de ocultamiento, que inserta el código QR en la fotografía portadora, y la red de revelado, que lo recupera desde la imagen contenedora. |
| **Marco Técnico** | Redes neuronales convolucionales basadas en la arquitectura de Baluja [1], adaptada para que la imagen secreta sea un QR de 1 canal. Entrenamiento conjunto con la función de pérdida L = MSE(C, C′) + β · MSE(S, S′), donde C es la portadora, C′ la contenedora, S el QR original, S′ el QR revelado y β controla el balance entre imperceptibilidad y calidad de recuperación. Optimizador Adam. Durante el entrenamiento se simula el redondeo a 8 bits para que el modelo funcione con imágenes PNG reales. |
| **Estructura General** | **Entrenamiento:** para cada lote (C, S) de M1: C′ = Ocultar(C, S); S′ = Revelar(C′); calcular L; retropropagar; guardar el mejor modelo según el conjunto de validación.  **Uso:** Ocultar(C, S) entrega C′, que se guarda como PNG; Revelar(C′) entrega S′.  **Entradas y salidas:** portadora (128×128×3) y QR (128×128×1) → contenedora (128×128×3); contenedora → QR revelado (128×128×1); archivo con los pesos del modelo entrenado.  **Extensión opcional:** con el mismo diseño y una salida de 3 canales, el modelo puede entrenarse para ocultar fotografías. |
| **Requisito Funcional** | RF1, RF2. |

**M3. Módulo de Lectura de QR (QR)**

|  |  |
| --- | --- |
| **Propósito** | Convertir el QR revelado, que es aproximado y tiene ruido, en un código QR limpio y decodificar el mensaje de texto. |
| **Marco Técnico** | Filtro de mediana para reducir ruido, umbralización global de Otsu [8], reescalado por vecino más cercano y adición de un margen blanco (zona de silencio). Decodificación con el detector de QR de OpenCV (cv2.QRCodeDetector). |
| **Estructura General** | **Entrada:** QR revelado S′ (128×128, escala de grises).  **Ejecución:** filtro de mediana → umbral de Otsu → margen blanco → ampliación ×4 → decodificación.  **Salida:** texto decodificado o aviso de «QR no legible», junto con la imagen del QR limpio. |
| **Requisito Funcional** | RF4. |

**M4. Módulo de Evaluación (EVA)**

|  |  |
| --- | --- |
| **Propósito** | Medir objetivamente el desempeño del sistema y verificar los criterios de aceptación definidos en la propuesta. |
| **Marco Técnico** | PSNR = 10·log₁₀(MAX² / MSE); índice de similitud estructural SSIM [9]; tasa de lectura de QR = QR cuyo texto decodificado coincide exactamente con el original / total de QR evaluados. Implementación con scikit-image y OpenCV. |
| **Estructura General** | **Entradas:** modelo entrenado (M2), conjunto de prueba (M1) y lector de QR (M3).  **Ejecución:** para cada par de prueba: ocultar, guardar y volver a cargar como PNG, calcular PSNR y SSIM, revelar, aplicar M3 y comparar el texto decodificado con el original.  **Salidas:** tabla con promedio y desviación de las métricas, gráficos, ejemplos de éxito y falla, e indicación de cumplimiento (PSNR ≥ 30 dB, SSIM ≥ 0,90, lectura ≥ 90 %). En la interfaz entrega las métricas de cada ejecución. |
| **Requisito Funcional** | RF5 (verifica el cumplimiento de RF1 y RF4). |

**M5. Módulo de Interfaz Web (WEB)**

|  |  |
| --- | --- |
| **Propósito** | Permitir que un usuario sin conocimientos técnicos use el sistema desde el navegador. |
| **Marco Técnico** | Aplicación web construida con Gradio (Python), ejecutable en un computador local o en Colab mediante un enlace público temporal. Carga el modelo entrenado de M2 y utiliza M1, M3 y M4. |
| **Estructura General** | **Pestaña «Ocultar»:** el usuario carga una fotografía y escribe un mensaje, que M1 convierte en QR; al presionar «Ocultar» obtiene la contenedora descargable en PNG y sus métricas (M4).  **Pestaña «Revelar»:** el usuario carga una contenedora; al presionar «Revelar» obtiene el QR revelado y el mensaje decodificado (M3). |
| **Requisito Funcional** | RF6 (integra RF1 a RF5). |