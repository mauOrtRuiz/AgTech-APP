let model;
const video = document.getElementById('webcam');
const label = document.getElementById('label');

async function app() {
    label.innerText = "Cargando Modelo IA...";
    
    try {
        // 1. Cargamos el modelo (Asegúrate que la carpeta se llame modelo_web)
        model = await tf.loadLayersModel('modelo_web/model.json');
        label.innerText = "Modelo Cargado. Abriendo Cámara...";

        // 2. Configuramos la cámara para celular
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" }, // Usa la cámara trasera
            audio: false
        });
        video.srcObject = stream;

        // 3. Bucle de predicción
        video.onloadedmetadata = () => {
            predict();
        };

    } catch (error) {
        console.error(error);
        label.innerText = "Error: No se encontró la carpeta 'modelo_web' o la cámara.";
    }
}

async function predict() {
    while (true) {
        const result = tf.tidy(() => {
            const img = tf.browser.fromPixels(video);
            const resized = tf.image.resizeBilinear(img, [224, 224]);
            const normalized = resized.div(255.0).expandDims(0);
            return model.predict(normalized);
        });

        const prediction = await result.data();
        const highestIndex = result.argMax(1).dataSync()[0];
        
        // Cambia estas etiquetas por las de tus plantas
        const clases = ["Sana", "Enferma Tipo A", "Enferma Tipo B"]; 
        label.innerText = `Resultado: ${clases[highestIndex] || 'Clase ' + highestIndex}`;

        result.dispose();
        await tf.nextFrame();
    }
}

app();