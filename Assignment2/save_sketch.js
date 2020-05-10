var capture;
let classifier;
let feature;
var name;
var prob;
let Lo = 100;
var inp;
var labels = new Array();
var inputs = new Array();

function GetResult(error, results) {
  if (error) {
    console.error(error);
  } else {
    console.log(results);
    name = results;
  }
}

function ModelReady() {
  console.log('Model is ready!');
}

function whileTraining(loss) {
  console.log("Loss is:" + loss);
  Lo = loss;
  if (loss == null) {
    classifier.classify(GetResult)
  }
}

function captureReady() {
  console.log('capture is ready!');
}

function myInputEvent() {
  classifier.numClasses = inp.value();
  if (inp.value() == 1)
    return;
  for (let i = 1; i <= inp.value(); i++) {
    na = createButton('category' + i);
    labelInp = createInput('category' + i + ' label');
    inputs[i] = labelInp;
    labels[i] = labelInp.value();
    na.mousePressed(function () {
      print(i)
      print(labels[i]);
      //classifier.addImage('category' + i);
      classifier.addImage(labels[i]);
    });
  }

  console.log('Array len ' + inputs.length);
  for (let i = 1; i < inputs.length; i++) {
    //console.log(inputs[i-1].value());
    inputs[i].input(function () {
      console.log(inputs[i].value());
      labels[i] = inputs[i].value();
    });
  }
}

function setup() {
  createCanvas(320, 250);
  capture = createCapture(VIDEO);
  capture.hide();
  feature = ml5.featureExtractor('MobileNet', ModelReady);
  classifier = feature.classification(capture, captureReady);

  inp = createInput('Categries');
  inp.position(0, 0);
  inp.input(myInputEvent);

  console.log('Array len ' + inputs.length);

  trainButton = createButton('Train');
  trainButton.position(200, 0);
  trainButton.mousePressed(function () {
    classifier.train(whileTraining);
  });

  saveButton = createButton('Save');
  saveButton.position(0, 25);
  saveButton.mousePressed(function () {
    classifier.save();
  });
}

function draw() {
  image(capture, 0, 50, width, height);
  // image(capture, 0, 50, width, width * capture.height / capture.width);
  if (Lo == null) {
    classifier.classify(GetResult)

    fill(255, 0, 0);
    textSize(30);
    text(name, 10, height - 20);
  }
}
