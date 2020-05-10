let mnist;
let inputs = [];
let show = 1;
let train_image;

const model = tf.sequential();

model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  kernelSize: 5,
  filters: 16,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling',
}));

model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2],
  strides: [2, 2]
}));

model.add(tf.layers.conv2d({
  kernelSize: 5,
  filters: 32,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling',
}));

model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2],
  strides: [2, 2]
}));

model.add(tf.layers.flatten());

model.add(tf.layers.dense({
  units: 128,
  activation: 'relu'
}));

model.add(tf.layers.dropout({
  rate: 0.2
}));

model.add(tf.layers.dense({
  units: 64,
  activation: 'relu'
}));

model.add(tf.layers.dropout({
  rate: 0.2
}));

model.add(tf.layers.dense({
  units: 10,
  // activation: 'softmax'
}));
const OPT = tf.train.adam(0.001)
const config = {
  optimizer: OPT,
  loss: tf.losses.softmaxCrossEntropy,
}
model.compile(config);


function setup() {
  train_image = createImage(28, 28);
  loadMNIST(function (data) {
    mnist = data;
    console.log(mnist);
  })

  // save model
  createCanvas(200, 200);
  saveButton = createButton('Save');
  saveButton.position(200, 0);
  saveButton.mousePressed(async function () {
    console.log('save model..')
    const saveResults = await model.save('downloads://my-model');
    console.log(saveResults);
  });
}

function draw() {
  if (mnist) {
    //train data
    inputs = tf.tensor2d(mnist.train_images);
    outputs_org = tf.tensor1d(mnist.train_labels);
    //one hot embedding
    outputs = tf.oneHot((outputs_org), 10);

    console.log(inputs.reshape([-1, 28, 28, 1]));

    //test data
    inputs_test = tf.tensor2d(mnist.test_images);
    outputs_test = tf.tensor1d(mnist.test_labels);

    async function train() {
      for (let i = 1; i <= 2; i++) {
        const h = await model.fit(inputs.reshape([-1, 28, 28, 1]), outputs, {
          batchSize: 64,
          epochs: 1
        });
        console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);

        output_tem = model.predict(inputs_test.reshape([-1, 28, 28, 1]));
        label = tf.argMax(output_tem, 1);
        // label.print();
        //tf.add(label, 1).print();
        tf.div(tf.sum(outputs_test.equal(label)), mnist.test_labels.length).print();
      }
    }

    train().then(() => {
      // model.predict(inputs).print();
      output_tem = model.predict(inputs_test.reshape([-1, 28, 28, 1]));
      label = tf.argMax(output_tem, 1);
      //tf.add(label, 1).print();
      tf.div(tf.sum(outputs_test.equal(label)), mnist.test_labels.length).print();
    })
    noLoop();
  }
}
