const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");
const iris = require("./iris.json");
const irisTesting = require("./iris-testing.json");

const normalizeData = (iris) => [iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width];
const normalizeOutput = ({ species }) => [
  species === "setosa" ? 1 : 0,
  species === "virginica" ? 1 : 0,
  species === "versicolor" ? 1 : 0,
];

const trainingData = tf.tensor2d(iris.map(normalizeData));
const outputData = tf.tensor2d(iris.map(normalizeOutput));
const testingData = tf.tensor2d(irisTesting.map(normalizeData));

const model = tf.sequential();

model.add(tf.layers.dense({
  inputShape: [4],
  activation: "sigmoid",
  units: 5
}));

model.add(tf.layers.dense({
  inputShape: [5],
  activation: "sigmoid",
  units: 3
}));

model.add(tf.layers.dense({
  activation: "sigmoid",
  units: 3
}));

model.compile({
  loss: "meanSquaredError",
  optimizer: tf.train.adam(.06)
});

const startTime = Date.now();

model.fit(trainingData, outputData, { epochs: 100 })
  .then(() => {
    console.log("Done in", Date.now() - startTime);
    model.predict(testingData).print();
  });
