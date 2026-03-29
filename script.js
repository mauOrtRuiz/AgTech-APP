async function app() {
    const video = document.getElementById('webcam');
    const label = document.getElementById('label');

    // 1. FORZAMOS LA CÁMARA PRIMERO (Para romper el ciclo de error)
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });
        video.srcObject = stream;
        video.play();
        label.innerText = "Cámara lista. Cargando IA...";
    } catch (e) {
        label.innerText = "Error: No se pudo acceder a la cámara.";
        return;
    }

    // 2. INTENTAMOS CARGAR EL MODELO SIN BYPASS
    try {
        // Usamos solo LayersModel (el formato correcto de tu archivo)
        const model = await tf.loadLayersModel('modelo_web/model.json');
        label.innerText = "IA de AgTech Online.";

        // 3. INFERENCIA
        setInterval(async () => {
            const result = tf.tidy(() => {
                const img = tf.browser.fromPixels(video);
                const resized = tf.image.resizeBilinear(img, [224, 224]);
                const normalized = resized.div(255.0).expandDims(0);
                return model.predict(normalized);
            });
            const prediction = await result.data();
            const highestIndex = result.argMax(1).dataSync()[0];
            label.innerText = `Detección: Clase ${highestIndex}`;
            result.dispose();
        }, 1000);

    } catch (error) {
        console.error("Error al cargar modelo:", error);
        // Si sale el error de InputLayer aquí, el JSON necesita edición manual.
        label.innerText = "Error de arquitectura en model.json. (InputLayer)";
    }
}

app();