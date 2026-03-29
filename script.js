// AgTech - Script de Inferencia Optimizado
window.onload = async function() {
    const video = document.getElementById('webcam');
    const label = document.getElementById('label');

    if (!label || !video) return;

    label.innerText = "Cargando cerebro de AgTech...";

    try {
        console.log("Intentando cargar el modelo...");
        
        // ESTRATEGIA 1: Carga estándar (LayersModel)
        let model;
        try {
            model = await tf.loadLayersModel('modelo_web/model.json');
        } catch (e) {
            console.log("Fallo LayersModel, intentando con GraphModel...");
            // ESTRATEGIA 2: Carga flexible (GraphModel) para evitar error de InputLayer
            model = await tf.loadGraphModel('modelo_web/model.json');
        }

        label.innerText = "Modelo cargado. Abriendo cámara...";

        // Configuración de cámara para celular (Cámara trasera)
        const constraints = {
            video: {
                facingMode: "environment",
                width: { ideal: 224 },
                height: { ideal: 224 }
            },
            audio: false
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;

        video.onloadedmetadata = () => {
            video.play();
            predecir(model, video, label);
        };

    } catch (err) {
        console.error("Error crítico:", err);
        label.innerText = "Error: El modelo no es compatible o falta la cámara.";
    }
};

async function predecir(model, video, label) {
    while (true) {
        const result = tf.tidy(() => {
            // Captura el frame de la cámara
            const img = tf.browser.fromPixels(video);
            
            // Redimensiona a 224x224 (esto soluciona el error de Input Shape)
            const resized = tf.image.resizeBilinear(img, [224, 224]);
            
            // Normaliza los valores de los píxeles (0 a 1)
            const casted = resized.cast('float32');
            const offset = tf.scalar(255.0);
            const normalized = casted.div(offset);
            
            // Añade la dimensión de "batch" [1, 224, 224, 3]
            return normalized.expandDims(0);
        });

        // Ejecuta la predicción
        const prediction = await model.predict(result).data();
        
        // Obtiene el índice de la clase con mayor probabilidad
        const highestIndex = prediction.indexOf(Math.max(...prediction));
        const probabilidad = (prediction[highestIndex] * 100).toFixed(1);
        
        // Muestra el resultado en pantalla
        // PUEDES CAMBIAR LOS NOMBRES DE LAS CLASES AQUÍ:
        const nombresClases = ["Sana", "Enferma A", "Enferma B"]; 
        const nombre = nombresClases[highestIndex] || `Clase ${highestIndex}`;
        
        label.innerText = `Diagnóstico: ${nombre} (${probabilidad}%)`;

        result.dispose(); // Libera memoria para que el celular no se trabe
        await tf.nextFrame(); // Espera al siguiente frame
    }
}