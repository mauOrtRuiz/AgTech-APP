// AgTech-APP: Sistema de detección de cultivos
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
                { video: { facingMode: "environment" } }, // Usa la cámara trasera del celular
                stream => {
                    video.srcObject = stream;
                    video.addEventListener('loadeddata', () => resolve(), false);
                },
                error => reject());
        } else {
            reject();
        }
    });
}

async function app() {
    console.log('Cargando modelo AgTech...');
    label.innerText = "Cargando IA... espera un momento.";

    try {
        // SOLUCIÓN AL ERROR DE INPUTLAYER:
        // Cargamos el modelo como GraphModel si LayersModel falla
        model = await tf.loadLayersModel('modelo_web/model.json').catch(async (err) => {
            console.log("Cargando como GraphModel por error de arquitectura...");
            return await tf.loadGraphModel('modelo_web/model.json');
        });

        label.innerText = "IA Lista. Abre la cámara...";
        await setupWebcam();
        video.play();

        while (true) {
            const result = tf.tidy(() => {
                // Captura y pre-procesamiento de la imagen
                const img = tf.browser.fromPixels(video);
                
                // Forzamos el tamaño 224x224 para evitar el error de InputLayer
                const resized = tf.image.resizeBilinear(img, [224, 224]);
                const casted = resized.cast('float32');
                const offset = tf.scalar(255.0);
                const normalized = casted.div(offset);
                const batched = normalized.expandDims(0);
                
                return model.predict(batched);
            });

            const prediction = await result.data();
            const highestIndex = result.argMax(1).dataSync()[0];
            const confidence = (Math.max(...prediction) * 100).toFixed(2);

            // Personaliza aquí los nombres de tus clases
            const clases = ["Sana", "Plaga Detectada", "Deficiencia Nutricional"];
            label.innerText = `Resultado: ${clases[highestIndex] || 'Clase ' + highestIndex} (${confidence}%)`;

            result.dispose();
            await tf.nextFrame();
        }
    } catch (error) {
        console.error(error);
        label.innerText = "Error crítico: Verifica la carpeta 'modelo_web' o los permisos de cámara.";
    }
}

// Iniciar la aplicación
app();