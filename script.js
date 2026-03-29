async function app() {
    const video = document.getElementById('webcam');
    const label = document.getElementById('label');

    // 1. Intentamos abrir la cámara primero (Paso independiente)
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        await video.play();
        console.log("Cámara iniciada correctamente");
    } catch (e) {
        label.innerText = "Error: No se detecta la cámara o no hay permisos.";
        return;
    }

    label.innerText = "Cámara OK. Analizando modelo...";

    // 2. Cargamos el modelo con un catch que nos diga EXACTAMENTE qué falla
    try {
        // Cargamos el modelo
        const model = await tf.loadLayersModel('modelo_web/model.json');
        label.innerText = "¡IA CONECTADA! Apunta a una planta.";

        // 3. Bucle de predicción simplificado
        setInterval(async () => {
            const result = tf.tidy(() => {
                const img = tf.browser.fromPixels(video);
                const resized = tf.image.resizeBilinear(img, [224, 224]);
                const normalized = resized.div(255.0).expandDims(0);
                return model.predict(normalized);
            });

            const prediction = await result.data();
            const id = result.argMax(1).dataSync()[0];
            label.innerText = `ID de Clase detectada: ${id}`;
            result.dispose();
        }, 1000);

    } catch (error) {
        console.error("DETALLE DEL ERROR:", error);
        label.innerText = "ERROR DE MODELO: " + error.message;
    }
}

app();