let model;
const video = document.getElementById('webcam');
const label = document.getElementById('label');

async function setupWebcam() {
    return new Promise((resolve, reject) => {
        const navigatorAny = navigator;
        navigator.getUserMedia = navigator.getUserMedia ||
            navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia || navigatorAny.msGetUserMedia;
        if (navigator.getUserMedia) {
            navigator.getUserMedia({ video: true },
                stream => {
                    video.srcObject = stream;
                    video.addEventListener('loadeddata', () => resolve(), false);
                },
                error => reject());
        } else {
            reject();
        }
    });
}

async function app() {
    console.log('Cargando modelo...');
    label.innerText = "Cargando IA de AgTech...";
    
    // Carga el modelo desde tu carpeta en el servidor
    model = await tf.loadLayersModel('modelo_web/model.json');
    
    console.log('Modelo cargado correctamente');
    label.innerText = "IA Lista. Apunta a una hoja.";

    await setupWebcam();

    while (true) {
        const result = tf.tidy(() => {
            const img = tf.browser.fromPixels(video);
            // ESTO CORRIGE EL ERROR DE INPUTLAYER:
            const resized = tf.image.resizeBilinear(img, [224, 224]);
            const casted = resized.cast('float32');
            const offset = tf.scalar(255.0);
            const normalized = casted.div(offset);
            const batched = normalized.expandDims(0);
            return model.predict(batched);
        });

        const prediction = await result.data();
        const highestIndex = result.argMax(1).dataSync()[0];
        
        // Aquí puedes poner los nombres de tus enfermedades según tu modelo
        label.innerText = `Detección: Clase ${highestIndex} (${(Math.max(...prediction) * 100).toFixed(2)}%)`;

        result.dispose();
        await tf.nextFrame();
    }
}

app();