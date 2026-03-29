// AgTech-APP: Sistema de detección de cultivos en tiempo real
let model;
const video = document.getElementById('webcam');
const label = document.getElementById('label');

async function setupWebcam() {
    return new Promise((resolve, reject) => {
        const navigatorAny = navigator;
        navigator.getUserMedia = navigator.getUserMedia ||
            navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia || navigatorAny.msGetUserMedia;
        
        if (navigator.getUserMedia) {
            navigator.getUserMedia(
                { video: { facingMode: "environment" } }, // Fuerza la cámara trasera del celular
                stream => {
                    video.srcObject = stream;
                    video.addEventListener('loadeddata', () => resolve(), false);
                },
                error => {
                    console.error("Error al acceder a la cámara:", error);
                    reject(error);
                }
            );
        } else {
            reject("Navegador no soporta getUserMedia");
        }
    });
}

async function app() {
    console.log('Iniciando AgTech AI...');
    label.innerText = "Cargando cerebro de la IA...";

    try {
        // SOLUCIÓN DEFINITIVA AL ERROR DE INPUTLAYER:
        // Intentamos cargar el modelo. Si falla por arquitectura, usamos loadGraphModel que es más flexible.
        try {
            model = await tf.loadLayersModel('modelo_web/model.json');
            console.log("Modelo cargado como LayersModel");
        } catch (e) {
            console.warn("Error de InputLayer detectado. Aplicando bypass con GraphModel...");
            model = await tf.loadGraphModel('modelo_web/model.json');
            console.log("Modelo cargado con éxito como GraphModel");
        }

        label.innerText = "IA Lista. Accediendo a cámara...";
        
        await setupWebcam();
        video.play();

        label.innerText = "Escaneando cultivo...";

        while (true) {
            const result = tf.tidy(() => {
                // Captura el frame actual de la cámara
                const img = tf.browser.fromPixels(video);
                
                // Redimensionamos a 224x224 (el tamaño estándar de MobileNet/AgTech)
                const resized = tf.image.resizeBilinear(img, [224, 224]);
                
                // Normalización de píxeles y creación de Tensor 4D
                const casted = resized.cast('float32');
                const offset = tf.scalar(255.0);
                const normalized = casted.div(offset);
                const batched = normalized.expandDims(0);
                
                return model.predict(batched);
            });

            const prediction = await result.data();
            const highestIndex = result.argMax(1).dataSync()[0];
            const confidence = (Math.max(...prediction) * 100).toFixed(1);

            // --- PERSONALIZA TUS CLASES AQUÍ ---
            const clases = ["Sana", "Plaga Detectada", "Deficiencia de Nutrientes"];
            const diagnostico = clases[highestIndex] || `Clase ${highestIndex}`;
            
            label.innerText = `Diagnóstico: ${diagnostico} (${confidence}%)`;

            result.dispose(); // Liberamos memoria para que el celular no se caliente
            await tf.nextFrame();
        }
    } catch (error) {
        console.error("Error crítico de ejecución:", error);
        label.innerText = "Error: Verifica la carpeta 'modelo_web' o los permisos de cámara.";
    }
}

// Iniciar la aplicación
app();