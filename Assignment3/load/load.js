let model;
let mnist;

async function loadmodel() {
  console.log('model loading...');

  // clear the model variable
  model = undefined;

  // load the model by using a HTTPS request
  model = await tf.loadLayersModel('../model/my-model.json');

  console.log('model loaded...')
}

function loadmnist() {
	console.log('mnist loading...');

	loadMNIST(function (data) {
			mnist = data;
			console.log(data);
		});
	console.log('mnist loaded...');
}

function setup() {
	loadmodel();

	loadmnist();
}

function draw() {
	if (mnist && model) {
		// test data
		inputs_test = tf.tensor2d(mnist.test_images);
    outputs_test = tf.tensor1d(mnist.test_labels);

    output_tmp = model.predict(inputs_test);
    label = tf.argMax(output_tmp, 1);
    tf.div(tf.sum(outputs_test.equal(label)), mnist.test_labels.length).print();

    // eval_result = tf.div(tf.sum(outputs_test.equal(label)), mnist.test_labels.length);
    // console.log(eval_result);

    noLoop();
	}
}
