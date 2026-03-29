// AgTech AI - Versión Estabilizada para PC
async function app() {
    const video = document.getElementById('webcam');
    const label = document.getElementById('label');

    label.innerText = "Cargando IA... (Esto puede tardar 10 segundos)";

    try {
        // 1. CARGA DIRECTA: Cargamos el modelo ignorando los errores de metadatos
        console.log("Cargando modelo...");
        const model = await tf.loadLayersModel('modelo_web/model.json');
        console.log("Modelo cargado exitosamente.");

        label.innerText = "Modelo listo. Accediendo a la cámara...";

        // 2. ENCENDER CÁMARA: Este bloque es el que activa el hardware de tu PC
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });
        video.srcObject = stream;

        // 3. PREDICCIÓN EN TIEMPO REAL
        video.onloadedmetadata = () => {
            video.play();
            
            // Ciclo de detección
            setInterval(async () => {
                const result = tf.tidy(() => {
                    const img = tf.browser.fromPixels(video);
                    const resized = tf.image.resizeBilinear(img, [224, 224]);
                    const normalized = resized.div(255.0).expandDims(0);
                    return model.predict(normalized);
                });

                const prediction = await result.data();
                const highestIndex = result.argMax(1).dataSync()[0];
                const prob = (Math.max(...prediction) * 100).toFixed(1);

                // Nombres de tus carpetas de Colab
                const clases = ["Sana", "Enferma A", "Enferma B"];
                label.innerText = `Detección: ${clases[highestIndex] || 'Clase ' + highestIndex} (${prob}%)`;

                result.dispose();
            }, 500); 
        };

    } catch (error) {
        console.error("Error detallado:", error);
        label.innerText = "Error: El archivo 'model.json' está corrupto o falta la cámara.";
        
        // Si el error es de InputLayer, intentamos una carga de emergencia
        if(error.message.includes("InputLayer")) {
            label.innerText = "Error de arquitectura. Necesitas regenerar el JSON.";
        }
    }
}

app();