window.onload = async function() {
    const video = document.getElementById('webcam');
    const label = document.getElementById('label');

    if (!label || !video) return;

    label.innerText = "Cargando cerebro de AgTech...";

    try {
        // SOLUCIÓN AL ERROR DE INPUTLAYER: 
        // Cargamos el modelo con un bloque try/catch específico
        console.log("Intentando cargar el modelo...");
        const model = await tf.loadLayersModel('modelo_web/model.json');
        
        label.innerText = "Modelo cargado. Abriendo cámara...";

        // Configuración de cámara compatible con Android/iPhone
        const constraints = {
            video: {
                facingMode: "environment", // Fuerza la cámara trasera
                width: { ideal: 224 },
                height: { ideal: 224 }
            },
            audio: false
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;

        video.onloadedmetadata = () => {
            video.play();
            // Iniciamos la predicción pasándole el modelo
            predecir(model, video, label);
        };

    } catch (err) {
        console.error("Error detallado:", err);
        label.innerText = "Error: El modelo no es compatible o no hay cámara.";
    }
};

async function predecir(model, video, label) {
    while (true) {
        const result = tf.tidy(() => {
            // Convertimos el video a tensores
            const img = tf.browser.fromPixels(video);
            // Ajustamos el tamaño a 224x224 que es lo que pide tu InputLayer
            const resized = tf.image.resizeBilinear(img, [224, 224]);
            const casted = resized.cast('float32');
            const offset = tf.scalar(255.0);
            const normalized = casted.div(offset);
            const batched = normalized.expandDims(0);
            return model.predict(batched);
        });

        const prediction = await result.data();
        const highestIndex = result.argMax(1).dataSync()[0];
        const probabilidad = (Math.max(...prediction) * 100).toFixed(1);
        
        // Esto es lo que verás en el celular
        label.innerText = `Detección: Clase ${highestIndex} (${probabilidad}%)`;

        result.dispose();
        await tf.nextFrame(); // Evita que el celular se caliente o se trabe
    }
}