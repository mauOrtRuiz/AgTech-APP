window.onload = async function() {
    const video = document.getElementById('webcam');
    const label = document.getElementById('label');

    if (!label || !video) return;

    label.innerText = "Iniciando AgTech en PC...";

    try {
        // 1. Cargamos el JSON manualmente para corregir el error de la capa de entrada
        const response = await fetch('modelo_web/model.json');
        const modelJson = await response.json();

        // PARCHE TÉCNICO: Si al modelo le falta la forma de entrada, se la inyectamos aquí
        if (modelJson.modelTopology && modelJson.modelTopology.model_config) {
            const layers = modelJson.modelTopology.model_config.config.layers;
            if (layers && layers[0] && layers[0].config) {
                layers[0].config.batch_input_shape = [null, 224, 224, 3];
            }
        }

        // 2. Cargamos el modelo corregido desde la memoria
        // Usamos una técnica de IO (Input/Output) para saltarnos el error de 'producer'
        const model = await tf.loadLayersModel(tf.io.fromMemory(
            modelJson.modelTopology,
            modelJson.format,
            modelJson.generatedBy,
            modelJson.convertedBy,
            modelJson.weightsManifest
        ));
        
        label.innerText = "IA Sincronizada. Accediendo a cámara...";

        // 3. Encender la cámara de la PC
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });
        video.srcObject = stream;

        video.onloadedmetadata = () => {
            video.play();
            // Iniciamos el bucle de detección
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

                // Cambia estos nombres por tus categorías reales
                const clases = ["Sana", "Enferma A", "Enferma B"];
                label.innerText = `Resultado: ${clases[highestIndex] || 'Clase ' + highestIndex} (${prob}%)`;

                result.dispose();
            }, 500); // Procesa cada 500ms para no saturar la PC
        };

    } catch (err) {
        console.error("Error detallado en PC:", err);
        label.innerText = "Error de arquitectura. Revisa la consola (F12).";
    }
};