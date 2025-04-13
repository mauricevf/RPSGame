import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const enableWebcamButton = document.getElementById("webcamButton")
const logButton = document.getElementById("logButton")

const video = document.getElementById("webcam")
const canvasElement = document.getElementById("output_canvas")
const canvasCtx = canvasElement.getContext("2d")

const drawUtils = new DrawingUtils(canvasCtx)
let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;
let nn;
let testdata;

fetch("./model/data.json")
    .then(response => response.json())
    .then(data => {
        testdata = data;
        createNeuralNetwork();
    });

const createNeuralNetwork = async () => {
    ml5.setBackend("webgl")
    nn = ml5.neuralNetwork({ task: 'classification', debug: true })
    const options = {
        model: "./model/model.json",
        metadata: "./model/model_meta.json",
        weights: "./model/model.weights.bin",
    }
    nn.load(options, () => {
        console.log("Neural network loaded successfully.");
        createHandLandmarker();
    });
}

let image = document.querySelector("#myimage")


/********************************************************************
// CREATE THE POSE DETECTOR
********************************************************************/
const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });
    console.log("model loaded, you can start webcam")

    enableWebcamButton.addEventListener("click", (e) => enableCam(e))
    logButton.addEventListener("click", (e) => classifyHand(e))
}

/********************************************************************
// START THE WEBCAM
********************************************************************/
async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.style.width = video.videoWidth;
            canvasElement.style.height = video.videoHeight;
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            document.querySelector(".videoView").style.height = video.videoHeight + "px";
            predictWebcam();
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

/********************************************************************
// START PREDICTIONS    
********************************************************************/
async function predictWebcam() {
    results = await handLandmarker.detectForVideo(video, performance.now())

    let hand = results.landmarks[0]
    if (hand) {
        let thumb = hand[4]
        image.style.transform = `translate(${video.videoWidth - thumb.x * video.videoWidth}px, ${thumb.y * video.videoHeight}px)`
    }

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    for (let hand of results.landmarks) {
        drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
        drawUtils.drawLandmarks(hand, { radius: 4, color: "#FF0000", lineWidth: 2 });
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam)
    }
}
function classifyHand() {
    let numbersonly = [];
    console.log("Classifying with neural network...");
    if (!results || !results.landmarks || results.landmarks.length === 0) {
        console.warn("No hand detected.");
        return;
    }
    let hand = results.landmarks[0];
    for (let point of hand) {
        numbersonly.push(point.x, point.y, point.z);
    }
    console.log(numbersonly)
    console.log("Hand landmarks (x, y, z):", numbersonly);

    console.log("Input to neural network:", numbersonly);
    console.log(testdata)
    nn.classify(numbersonly, (results) => {
        console.log(`Ik denk dat dit een ${results[0].label} is`);
    });
}



/********************************************************************
// LOG HAND COORDINATES IN THE CONSOLE
********************************************************************/
function logAllHands() {
    for (let hand of results.landmarks) {
        // console.log(hand)
        console.log(hand[4])
    }
}

/********************************************************************
// START THE APP
********************************************************************/
if (navigator.mediaDevices?.getUserMedia) {
    createNeuralNetwork()
}