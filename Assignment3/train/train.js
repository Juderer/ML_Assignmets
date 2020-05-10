let mnist;
let inputs = [];
let show = 1;
let train_image;

const model = tf.sequential();
model.add(tf.layers.dense({
  units: 64,
  inputShape: [784],
  activation: 'relu'
}));

model.add(tf.layers.dense({
  units: 10,
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

train_index = 0;//第几个图片

function draw() {
  if (mnist) {
    //train data
    inputs = tf.tensor2d(mnist.train_images);
    outputs_org = tf.tensor1d(mnist.train_labels);
    //one hot embedding
    outputs = tf.oneHot((outputs_org), 10);

    //test data
    inputs_test = tf.tensor2d(mnist.test_images);
    outputs_test = tf.tensor1d(mnist.test_labels);

    async function train() {
      for (let i = 1; i < 5; i++) {
        const h = await model.fit(inputs, outputs, {
          batchSize: 64,
          epochs: 1
        });
        console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
      }
    }

//	    //保存
//	     async function train() {
//            for (let i = 1; i <= 100; ++i) {
//                const h = await model.fit(inputs, 		outputs,
//               {
//                    batchSize: 1000,
//                    epochs: 1
//                });
//                console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
//            }
//            const saveResults = await 		model.save('indexeddb://my-model-1');
//        }


    train().then(() => {
      // model.predict(inputs).print();
      output_tem = model.predict(inputs_test);
      label = tf.argMax(output_tem, 1);
      //tf.add(label, 1).print();
      tf.div(tf.sum(outputs_test.equal(label)), mnist.test_labels.length).print();
    })
    noLoop();
  }

//    let inputs = [];
//    if (mnist) {
//        train_image.loadPixels();
// 
//        for (let i = 0; i < 784; i++) {
//            let bright = mnist.train_images[train_index][i];
//            inputs[i] = bright / 255;
//            if (mnist) {
//                let index = i * 4;
//                train_image.pixels[index + 0] = bright;
//                train_image.pixels[index + 1] = bright;
//                train_image.pixels[index + 2] = bright;
//                train_image.pixels[index + 3] = 255;
//            }
//        }
//        if (mnist) {
//            train_image.updatePixels();
//            image(train_image, 0, 0, 200, 200);
//        }
//    }
}
