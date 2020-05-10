let model;
let user_digit;
let guess_value;
let pre_inputs = [];

let user_has_drawing = false;

async function loadmodel() {
  // clear the model variable
  model = undefined;
  // load the model by using a HTTPS request
  model = await tf.loadLayersModel('../model/my-model.json');

  console.log('model loaded..')
}

async function loadmnist() {
	loadMNIST(function (data) {
			mnist = data;
			console.log(data);
		});

	console.log('mnist loaded...');
}

function setup() {
	// load model
	loadmodel();
	// load mnist dataset
	loadmnist();

	createCanvas(200, 200).parent('container');

	user_digit = createGraphics(200, 200);
  user_digit.pixelDensity(1);

  user_guess_ele = select('#user_guess');
}

function guessUserDigit() {
	let img = user_digit.get();

  if(!user_has_drawing) {
    user_guess_ele.html('_');
  }

  let inputs = [];
  img.resize(28, 28);
  img.loadPixels();
  for (let i = 0; i < 784; i++) {
    inputs[i] = img.pixels[i * 4] / 255;
  }
  if (pre_inputs) {
  	console.log(pre_inputs==inputs);
	}

  let user_input = tf.tensor(inputs).reshape([-1, 28, 28, 1]);

  // console.log(user_input.data());

	let predictions = model.predict(user_input);
	let result = tf.argMax(predictions, 1).toString();
	console.log(result.toString());

	let arr = result.split('');
	guess_value = arr[arr.length - 2]

	user_guess_ele.html(guess_value);
}

function keyPressed() {
  if (key == ' ') {
    user_has_drawing = false;
    user_digit.background(0);
  }
}

function draw() {
	if (model) {
		background(0);

		guessUserDigit();

		image(user_digit, 0, 0);

		if (mouseIsPressed) {
			user_has_drawing = true;
			user_digit.stroke(255);
			user_digit.strokeWeight(16);
			user_digit.line(mouseX, mouseY, pmouseX, pmouseY);
		}
		// noLoop();
	}
}
