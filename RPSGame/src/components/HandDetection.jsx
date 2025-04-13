import { useEffect, useRef, useState } from 'react';
import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const HandDetection = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const imageRef = useRef(null);
    const drawUtilsRef = useRef(null);

    const [handLandmarker, setHandLandmarker] = useState(null);
    const [nn, setNn] = useState(null);
    const [results, setResults] = useState(null);
    const [testdata, setTestdata] = useState(null);
    const [webcamRunning, setWebcamRunning] = useState(false);
    const [predictionMessage, setPredictionMessage] = useState("");
    const [cameraError, setCameraError] = useState("");
    const [modelsReady, setModelsReady] = useState(false);

    useEffect(() => {
        fetch("/model/data.json")
            .then(res => res.json())
            .then(data => {
                setTestdata(data);
                createNeuralNetwork();
            });
    }, []);

    const createNeuralNetwork = async () => {
        await window.ml5.setBackend("webgl");
        await window.ml5.tf.ready();

        const nnInstance = window.ml5.neuralNetwork({ task: 'classification', debug: true });

        const options = {
            model: "/model/model.json",
            metadata: "/model/model_meta.json",
            weights: "/model/model.weights.bin",
        };

        nnInstance.load(options, () => {
            setNn(nnInstance);
            createHandLandmarker();
        });
    };

    const createHandLandmarker = async () => {
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
        const detector = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 2
        });

        setHandLandmarker(detector);
        setModelsReady(true);

        if (canvasRef.current) {
            drawUtilsRef.current = new DrawingUtils(canvasRef.current.getContext("2d"));
        }
    };

    const enableCam = async () => {
        if (!handLandmarker || webcamRunning || videoRef.current?.srcObject) return;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            const video = videoRef.current;

            if (video) {
                video.srcObject = stream;
                setWebcamRunning(true);
                setCameraError("");

                video.addEventListener('loadeddata', () => {
                    const canvas = canvasRef.current;
                    if (canvas) {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        predictWebcam();
                    }
                });
            }
        } catch (err) {
            console.error("Error accessing webcam:", err);
            setCameraError("Could not access the webcam. Please allow camera access or check if another app is using it.");
        }
    };

    const predictWebcam = async () => {
        if (!handLandmarker || !videoRef.current) return;

        const currentResults = await handLandmarker.detectForVideo(videoRef.current, performance.now());
        setResults(currentResults);

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (currentResults.landmarks.length > 0) {
            const hand = currentResults.landmarks[0];
            const thumb = hand[4];
            const img = imageRef.current;

            if (img) {
                img.style.transform = `translate(${canvas.width - thumb.x * canvas.width}px, ${thumb.y * canvas.height}px)`;
            }
        }

        if (webcamRunning) {
            window.requestAnimationFrame(predictWebcam);
        }
    };

    const classifyHand = () => {
        if (!results || results.landmarks.length === 0) return;
        const hand = results.landmarks[0];
        const input = [];

        for (let point of hand) {
            input.push(point.x, point.y, point.z);
        }

        if (nn) {
            nn.classify(input, (results) => {
                const prediction = results[0].label;
                setPredictionMessage(`I think this is a ${prediction}`);
            });
        }
    };

    return (
        <div className="container">
            <div className="header">
                <h1>Hand Gesture Detection</h1>
                <p>Click the button to start the webcam and detect hand gestures.</p>
            </div>

            <div className="controls">
                <button
                    onClick={enableCam}
                    className="btn start-btn"
                    disabled={!modelsReady}
                >
                    {modelsReady ? "Start Webcam" : "Loading Models..."}
                </button>

                <button
                    onClick={classifyHand}
                    className="btn classify-btn"
                    disabled={!results || !modelsReady}
                >
                    Classify Hand Gesture
                </button>
            </div>

            {/* Error or Prediction Message Display */}
            {cameraError && (
                <div className="error-message" style={{ color: 'red', marginTop: '10px' }}>
                    {cameraError}
                </div>
            )}
            {predictionMessage && !cameraError && (
                <div className="prediction-message">
                    {predictionMessage}
                </div>
            )}

            <div className="video-container" style={{ position: "relative" }}>
                <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    className="video"
                    style={{ width: "100%", maxWidth: "640px" }}
                />
                <canvas
                    ref={canvasRef}
                    className="canvas"
                    style={{ position: "absolute", top: 0, left: 0 }}
                />
                <div
                    ref={imageRef}
                    className="hand-indicator"
                    style={{
                        position: "absolute",
                        width: "20px",
                        height: "20px",
                        backgroundColor: "red",
                        borderRadius: "50%",
                        pointerEvents: "none"
                    }}
                ></div>
            </div>
        </div>
    );
};

export default HandDetection;
