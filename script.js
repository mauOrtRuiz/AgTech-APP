let model;
const video = document.getElementById('webcam');
const label = document.getElementById('label');

async function iniciarIA() {
    label.innerText = "Cargando cerebro de AgTech...";

    try {
        // 1. Cargar el modelo corregido
        // Importante: La ruta debe coincidir con tu carpeta en GitHub
        model = await tf.loadLayersModel('modelo_web/model.json');
        console.log("✅ Modelo cargado exitosamente");
        
        label.innerText = "IA Lista. Accediendo a cámara...";

        // 2. Abrir la cámara (Funciona en PC y Celular)
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" }, // Cámara trasera en celular
            audio: false
        });
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            video.play();
            predecir(); // Iniciar el bucle de detección
        };

    } catch (error) {
        console.error("Error inicializando:", error);
        label.innerText = "Error: " + error.message;
    }
}

async function predecir() {
    // Definimos las clases exactas de tu entrenamiento
    // Si tienes 38 clases, puedes poner los nombres aquí en orden
    const clases = ["Sana", "Plaga A", "Plaga B", "Deficiencia"]; 

    while (true) {
        const result = tf.tidy(() => {
            // Capturar frame de la webcam
            const img = tf.browser.fromPixels(video);
            
            // Pre-procesamiento (Igual al entrenamiento en Colab)
            const resized = tf.image.resizeBilinear(img, [224, 224]);
            const normalized = resized.div(255.0).expandDims(0);
            
            return model.predict(normalized);
        });

        const prediction = await result.data();
        const highestIndex = result.argMax(1).dataSync()[0];
        const confianza = (Math.max(...prediction) * 100).toFixed(1);

        // Mostrar resultado (Si el índice supera el nombre en el array, pone Clase X)
        const nombreClase = clases[highestIndex] || `Clase ${highestIndex}`;
        label.innerText = `${nombreClase} (${confianza}%)`;

        result.dispose();
        await tf.nextFrame(); // Esperar al siguiente frame para no saturar
    }
}

iniciarIA();