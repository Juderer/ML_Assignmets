var capture;
let classifier;
let feature;
var name;
var loaded = 0;

function GetResult(error, results) {
  if (error) {
    console.error(error);
  } else {
    console.log(results);
    name = results;
  }
}

function modelReady() {
  console.log('Model is ready!');
  classifier.load('model.json', customModelReady);
}

function customModelReady() {
	console.log('Custom Model is ready');
	loaded = 1;
}

function captureReady() {
  console.log('Capture is ready!');
}

function setup() {
  createCanvas(320, 250);
  capture = createCapture(VIDEO);
  capture.hide();
  feature = ml5.featureExtractor('MobileNet', modelReady);
  classifier = feature.classification(capture, captureReady);
}

function draw() {
  image(capture, 0, 0, width, height);

  if (loaded) {
		classifier.classify(GetResult);

		fill(255, 0, 0);
		textSize(30);
		text(name, 10, height - 20);
	}
}
