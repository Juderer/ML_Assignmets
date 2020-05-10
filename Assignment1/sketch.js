// Copyright (c) 2019 ml5
//
// Creating a regression extracting features of MobileNet. Build with p5js.

let featureExtractorX;    // x-axis feature extractor
let featureExtractorY;    // y-axis feature extractor
let regressorX;
let regressorY;
let video;
let loss;
let sliderX;
let sliderY;
let samples = 0;
let positionX ;
let positionY ;

function setup() {
  createCanvas(680, 480);
  // Create a video element
  video = createCapture(VIDEO);
  // Append it to the videoContainer DOM element
  video.hide();
  // Initializes the rectangle position
  positionX = width / 2;
  positionY = height / 2;

  // Extract the features from MobileNet
  featureExtractorX = ml5.featureExtractor('MobileNet', modelReady);
  featureExtractorY = ml5.featureExtractor('MobileNet', modelReady);

  // Create a new regressor using those features and give the video we want to use
  regressorX = featureExtractorX.regression(video, videoReady);
  regressorY = featureExtractorY.regression(video, videoReady);

  // Create the UI buttons
  setupButtons();
}

function draw() {
  image(video, 0, 0, width, height);
  noStroke();
  fill(255, 0, 0);    // red
  rectMode(CENTER);
  rect(positionX, positionY, 50, 50);
}

// A function to be called when the model has been loaded
function modelReady() {
  select('#modelStatus').html('Model loaded!');
}

// A function to be called when the video has loaded
function videoReady() {
  select('#videoStatus').html('Video ready!');
}

// Classify the current frame.
function predict() {
  regressorX.predict(gotXResults);
  regressorY.predict(gotYResults);
}

// A util function to create UI buttons
function setupButtons() {
  sliderX = select('#sliderX');
  sliderY = select('#sliderY');
  
  select('#addSample').mousePressed(function() {
    console.log('X label: ' + sliderX.value());
    regressorX.addImage(sliderX.value());

    console.log('Y label: ' + sliderY.value());
    regressorY.addImage(sliderY.value());

    select('#amountOfSamples').html(samples++);
  });

  // Train Button
  select('#train').mousePressed(function() {
    regressorX.train(function(lossValue) {
      if (lossValue) {
        loss = lossValue;
        select('#lossX').html('XLoss: ' + loss);
      } else {
        select('#lossX').html('Done Training! Final XLoss: ' + loss);
      }
    });

    regressorY.train(function(lossValue) {
      if (lossValue) {
        loss = lossValue;
        select('#lossY').html('YLoss: ' + loss);
      } else {
        select('#lossY').html('Done Training! Final YLoss: ' + loss);
      }
    });
  });

  // Predict Button
  select('#buttonPredict').mousePressed(predict);
}

// Show the results
function gotXResults(err, result) {
  if (err) {
    console.error(err);
  }
  if (result && result.value) {
    positionX = map(result.value, 0, 1, 0, width);
    sliderX.value(result.value);
    // predict();
  }
}

function gotYResults(err, result) {
  if (err) {
    console.error(err);
  }
  if (result && result.value) {
    positionY = map(result.value, 0, 1, 0, height);
    sliderY.value(result.value);
    predict();
  }
}
