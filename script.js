// Esperamos a que toda la página cargue antes de buscar los elementos
window.onload = async function() {
    const video = document.getElementById('webcam');
    const label = document.getElementById('label');

    // Verificamos si los elementos existen para evitar el error de "null"
    if (!label || !video) {
        console.error("No se encontraron los elementos 'webcam' o 'label' en el HTML");
        return;
    }

    label.innerText = "Cargando IA de AgTech...";

    try {
        // Cargar el modelo
        const model = await tf.loadLayersModel('modelo_web/model.json');
        label.innerText = "Modelo cargado. Abriendo cámara...";

        // Configurar cámara
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" },
            audio: false
        });
        video.srcObject = stream;

        // Bucle de predicción
        video.onloadedmetadata = () => {
            setInterval(async () => {
                const result = tf.tidy(() => {
                    const img = tf.browser.fromPixels(video);
                    const resized = tf.image.resizeBilinear(img, [224, 224]);
                    const normalized = resized.div(255.0).expandDims(0);
                    return model.predict(normalized);
                });

                const prediction = await result.data();
                const highestIndex = result.argMax(1).dataSync()[0];
                
                // Actualizamos el texto con el resultado
                label.innerText = `Detección: Clase ${highestIndex} (${(Math.max(...prediction) * 100).toFixed(2)}%)`;
                
                result.dispose();
            }, 500); // Procesa cada medio segundo para no saturar el celular
        };

    } catch (error) {
        console.error(error);
        label.innerText = "Error al iniciar: Revisa cámara o carpeta modelo_web";
    }
};