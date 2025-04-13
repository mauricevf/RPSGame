let testdata;
let nn;
let correct = 0
let i = 0;

// Load the test data and start the neural network
fetch("./test.json")
    .then(response => response.json())
    .then(data => {
        testdata = data;
        createNeuralNetwork();
    });

function createNeuralNetwork() {
    console.log("json loaded " + testdata.length)
    ml5.setBackend("webgl");
    nn = ml5.neuralNetwork({ task: 'classification', debug: true });

    const options = {
        model: "./model/model.json",
        metadata: "./model/model_meta.json",
        weights: "./model/model.weights.bin",
    };

    nn.load(options, showLabels);
}

function showLabels() {
    // console.log(nn)
    console.log('Starting Testing');
    for (let testpose of testdata) {
        console.log(testpose);
    }
    startTesting()
}

function startTesting() {
    let testpose = testdata[i]
    nn.classify(testpose.data, (results) => {
        if (results[0].label === testpose.label) {
            console.log(`${testpose.label} correct voorspeld`)
            correct++
        }
        // console.log(results)
        // console.log(testdata)
        // console.log(`I think it is a ${results[i].label} and in reality it is a ${testpose.label}`);
        i++;
        if (i < testdata.length) {
            startTesting();
        } else {
            const accuracy = correct / testdata.length
            console.log(accuracy * 100)
        }
    });
}

