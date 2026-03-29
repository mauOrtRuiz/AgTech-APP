// AgTech - Versión con Activación Manual
const video = document.getElementById('webcam');
const label = document.getElementById('label');

// Creamos un botón dinámicamente para forzar el permiso
const btn = document.createElement("button");
btn.innerText = "INICIAR CÁMARA AGTECH";
btn.style = "padding: 15px; background: #2e7d32; color: white; border: none; border-radius: 8px; font-size: 16px; margin: 20px;";
document.body.appendChild(btn);

btn.onclick = async () => {
    btn.innerText = "Cargando...";
    try {
        // 1. Pedir permiso de cámara tras el CLICK
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" },
            audio: false
        });
        video.srcObject = stream;
        video.play();
        
        btn.style.display = "none"; // Escondemos el botón si funciona
        label.innerText = "Cámara lista. Cargando IA...";

        // 2. Cargar el modelo
        const model = await tf.loadLayersModel('modelo_web/model.json');
        label.innerText = "IA de AgTech Online.";

        // 3. Empezar a predecir
        predecir(model);

    } catch (error) {
        console.error(error);
        label.innerText = "Error: No diste permiso o la cámara está ocupada.";
        btn.innerText = "REINTENTAR PERMISOS";
    }
};

async function predecir(model) {
    setInterval(async () => {
        const result = tf.tidy(() => {
            const img = tf.browser.fromPixels(video);
            const resized = tf.image.resizeBilinear(img, [224, 224]);
            const normalized = resized.div(255.0).expandDims(0);
            return model.predict(normalized);
        });
        const prediction = await result.data();
        const highestIndex = result.argMax(1).dataSync()[0];
        label.innerText = `Detección: Clase ${highestIndex} (${(Math.max(...prediction)*100).toFixed(1)}%)`;
        result.dispose();
    }, 500);
}