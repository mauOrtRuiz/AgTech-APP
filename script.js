// Esta línea es el "escudo": espera a que el HTML esté cargado al 100%
window.onload = async function() {
    const video = document.getElementById('webcam');
    const label = document.getElementById('label');

    // Verificación de seguridad
    if (!label) {
        console.error("No se encontró el elemento con ID 'label' en el HTML.");
        return;
    }

    label.innerText = "Cargando cerebro de AgTech...";

    try {
        // 1. Cargar el modelo desde tu carpeta
        const model = await tf.loadLayersModel('modelo_web/model.json');
        label.innerText = "Modelo cargado. Iniciando cámara...";

        // 2. Encender la cámara (trasera preferentemente)
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" },
            audio: false
        });
        video.srcObject = stream;

        // 3. Empezar a predecir cuando el video esté listo
        video.onloadedmetadata = () => {
            video.play();
            predict(model, video, label);
        };

    } catch (err) {
        console.error(err);
        label.innerText = "Error: Revisa permisos de cámara o carpeta modelo_web";
    }
};

async function predict(model, video, label) {
    while (true) {
        const result = tf.tidy(() => {
            const img = tf.browser.fromPixels(video);
            const resized = tf.image.resizeBilinear(img, [224, 224]);
            const normalized = resized.div(255.0).expandDims(0);
            return model.predict(normalized);
        });

        const prediction = await result.data();
        const highestIndex = result.argMax(1).dataSync()[0];
        
        // Ajusta los nombres según tus clases de AgTech
        label.innerText = `Detección: Clase ${highestIndex} (${(Math.max(...prediction) * 100).toFixed(1)}%)`;

        result.dispose();
        await tf.nextFrame(); // No satura el procesador del celular
    }
}