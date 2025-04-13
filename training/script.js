let posedata;

fetch('./data.json')
    .then(response => response.json())
    .then(data => {
        posedata = data;
    })
    .then(() => {
        startTraining();
    })
    .catch(error => console.error('Error loading JSON:', error));

let nn


function startTraining(){

    ml5.setBackend('webgl')
    nn = ml5.neuralNetwork({task: 'classification', debug: true})

    // nn.addData([3,3,3,4], {label: "Dog"})

    for(let pose of posedata){
        console.log(pose);
        nn.addData(pose.data, {label: pose.label})
    }

    nn.normalizeData()
    nn.train({epochs:30}, finishedTraining)
}

function finishedTraining(){
    console.log('finished training')
    nn.save()

    let demopose = posedata[10].data;
    nn.classify(demopose, (results) => {
        console.log("I think this is a...")
        console.log(results[0].label)
        console.log(` ${(results[0].confidence.toFixed(2) * 100)}% Sure`)
    })
}
