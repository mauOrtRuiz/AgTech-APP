let model;
const video = document.getElementById('webcam');
const label = document.getElementById('label');

async function app() {
    label.innerText = "Cargando cerebro de AgTech...";

    try {
        // SOLUCIÓN MAESTRA: Si falla como LayersModel, lo cargamos como GraphModel
        // Esto ignora el error de 'InputLayer' que te está saliendo
        model = await tf.loadLayersModel('modelo_web/model.json').catch(async (err) => {
            console.log("Capa de entrada no detectada, activando modo compatibilidad...");
            return await tf.loadGraphModel('modelo_web/model.json');
        });

        label.innerText = "IA Lista. Iniciando cámara...";

        // Configuración de cámara para celular (Cámara trasera)
        const constraints = {
            video: { facingMode: "environment" },
            audio: false
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;

        video.onloadedmetadata = () => {
            video.play();
            predecir();
        };

    } catch (error) {
        console.error(error);
        label.innerText = "Error: Revisa la carpeta 'modelo_web' o los permisos de cámara.";
    }
}

async function predecir() {
    while (true) {
        const result = tf.tidy(() => {
            const img = tf.browser.fromPixels(video);
            // Forzamos el tamaño que el modelo espera (224x224)
            const resized = tf.image.resizeBilinear(img, [224, 224]);
            const casted = resized.cast('float32');
            const normalized = casted.div(255.0);
            const batched = normalized.expandDims(0);
            return model.predict(batched);
        });

        const prediction = await result.data();
        const highestIndex = result.argMax(1).dataSync()[0];
        const porcentaje = (Math.max(...prediction) * 100).toFixed(1);

        // AQUÍ PUEDES PONER TUS CLASES:
        const clases = ["Sana", "Plaga Detectada", "Deficiencia"];
        label.innerText = `Resultado: ${clases[highestIndex] || 'Clase ' + highestIndex} (${porcentaje}%)`;

        result.dispose();
        await tf.nextFrame();
    }
}

app();