// initialising global variables
MODEL_URL1 = "saved_models/tfjs/Final_Recall/model.json";
MODEL_URL2 = "saved_models/tfjs/model23/model.json";

MODEL_URL = MODEL_URL1;

LABELS = {
  0: "NORMAL",
  1: "PNEUMONIA",
};
// RESPONSES to give based on the confidence of the prediction
const RESPONSES = [
  "Our diagnosis indicates a very low probability of pneumonia. You are likely healthy, but seek expert medical help if you want a full diagnosis.",
  "Our diagnosis indicates a low probability of pneumonia. You may have some mild symptoms or other conditions that are not related to pneumonia. You should monitor your health and consult a doctor if you feel worse.",
  "Our diagnosis indicates a moderate probability of pneumonia. You may have some signs of pneumonia or other respiratory infections. You should seek medical advice and get tested for pneumonia.",
  "Our diagnosis indicates a high probability of pneumonia. You may have severe symptoms or complications of pneumonia. We advise that you see a doctor as soon as possible and get tested and treated for pneumonia.",
  "Our diagnosis indicates a very high probability of pneumonia. Due to the possibly high severity of the condition, we advise that you seek medical attention as soon as possible.",
];

// tresholds to determine confidence levels in the prediction.
// predictions are between [0,1], the algorithm is confident in the prediction when close to either extreme
const responseThresholds = [0.1, 0.3, 0.5, 0.7, 0.9];

IMG_SIZE = 150;

// initialising the model
let model;
// loading the model
loadModel(MODEL_URL);

function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $("#img-file-preview").attr("src", e.target.result);
    };

    reader.readAsDataURL(input.files[0]);
  }
}
async function loadModel(MODEL_URL) {
  model = await tf.loadLayersModel(MODEL_URL);
  if (model) {
    console.log("Model loaded successfully");
    console.log(model.summary());
  }
  //   return model;
}

async function predict(imgElement) {
  console.log("predicting...");

  async function to_tfGrayscaleTensor(imgElement) {
    // Function to convert HTMLImageElement to TensorFlow.js Tensor
    async function imageElementToGrayscaleTensor(imgElement) {
      return tf.tidy(() => {
        // Create a canvas element
        const canvas = document.createElement("canvas");
        canvas.width = imgElement.width;
        canvas.height = imgElement.height;

        // Draw the image onto the canvas
        const ctx = canvas.getContext("2d");
        ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

        // Get the image data from the canvas
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        // Convert RGB image to grayscale using the luminance formula
        const data = new Float32Array(imageData.width * imageData.height);
        for (let i = 0; i < data.length; ++i) {
          const offset = i * 4;
          const r = imageData.data[offset];
          const g = imageData.data[offset + 1];
          const b = imageData.data[offset + 2];
          data[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        }

        // Create a 3D tensor from the grayscale data
        const tensor = tf.tensor3d(data, [canvas.height, canvas.width, 1]);

        // Adding batch dimension for the conv_2d input layer
        return tensor.expandDims(0);
      });
    }

    // Convert HTMLImageElement to grayscale TensorFlow.js Tensor
    const grayscaleTensor = await imageElementToGrayscaleTensor(imgElement);
    console.log(grayscaleTensor);
    return grayscaleTensor;
  }

  let imgTensor = await to_tfGrayscaleTensor(imgElement);
  console.log(`tensor: ${imgTensor}`);

  //   PREPROCESSING
  // RESIZING and NORMALIZING
  imgTensor = tf.image.resizeBilinear(imgTensor, [IMG_SIZE, IMG_SIZE]);
  imgTensor = imgTensor.div(tf.scalar(255));
  //   singleImagePlot(imgTensor);

  // predict the class based on the model. returns a tf.js tensor
  let prediction = model.predict(imgTensor);

  // converting the tf.js tensor to an array
  //   only with numerical value for the prediction.
  const predictionsArray = prediction.dataSync();
  let predictedProbability = predictionsArray[0];

  // Logging the prediction for debugging
  console.log(`Prediction Class: ${predictedProbability}`);
  return predictedProbability;
}

async function diagnosisResult(predictedProbability) {
  // sets the DOM elements to give feedback according to the prediction.
  const feedbackMessage = document.querySelector("#feedback-message");
  const diagnosisScore = document.querySelector("#diagnosis-score");

  var feedbackIndex = responseThresholds.reverse().findIndex(function (threshold) {
    return predictedProbability < threshold;
  });

  console.log(`feedbackIndex: ${feedbackIndex}, predicted: ${predictedProbability}`)
  if (feedbackIndex === -1) {
    feedbackIndex = RESPONSES.length - 1;
  }
  var feedback = RESPONSES[feedbackIndex];

  // Display the feedback message and download link
  feedbackMessage.textContent = feedback;
  feedbackMessage.classList.add("bg-dark");
  feedbackMessage.classList.add("p-5");
  feedbackMessage.classList.add("rounded");
  feedbackMessage.classList.add("text-light");
  diagnosisScore.textContent = "";
}

function submitForm(event) {
  console.log("submitting image...");
  event.preventDefault();
  var fileInput = document.getElementById("image-file");
  var file = fileInput.files[0];

  var reader = new FileReader();
  reader.addEventListener("load", function () {
    // Get the data URL of the file content
    var dataURL = reader.result;
    // Create an image object
    var image = new Image();
    // Add an onload event listener
    image.onload = async function () {
      console.log(image.width, image.height);
      let predictedProbability = await predict(image);
      await diagnosisResult(predictedProbability);
    };
    image.src = dataURL;
  });
  // Read the file content as a data URL
  reader.readAsDataURL(file);
}
var form = document.getElementById("diagnosis-form");
form.addEventListener("submit", submitForm);

async function singleImagePlot(image) {
  // useless
  const canvas = document.createElement("canvas");
  canvas.width = 28;
  canvas.height = 28;
  canvas.style = "margin: 4px;";
  await tf.browser.toPixels(image, canvas);
  return canvas;
}
