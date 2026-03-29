let model;
const video = document.getElementById('camera');
const resultText = document.getElementById('result');

// 1. Cargar el modelo de TensorFlow.js
async function loadModel() {
    resultText.innerText = "Cargando Inteligencia AgTech...";
    // Asegúrate que la carpeta 'modelo_web' esté junto a este archivo
    model = await tf.loadLayersModel('modelo_web/model.json');
    resultText.innerText = "AgTech Listo. Apunta a una hoja.";
    setupCamera();
}

// 2. Configurar la cámara del celular
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" }, // Usa la cámara trasera
        audio: false
    });
    video.srcObject = stream;
}

// 3. Lógica de Predicción
async function predict() {
    if (!model) return;

    // Pre-procesamiento de la imagen (igual que en Colab)
    let tensor = tf.browser.fromPixels(video)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();

    const predictions = await model.predict(tensor).data();
    const maxPrediction = Math.max(...predictions);
    const index = predictions.indexOf(maxPrediction);

    // Aquí deberías tener una lista de tus etiquetas (labels)
    // Por ahora mostramos el índice y la confianza
    resultText.innerText = `Detección: Clase ${index} \nConfianza: ${(maxPrediction * 100).toFixed(2)}%`;
}

loadModel();