MODEL_URL = "saved_models/tfjs/model23/model.json";
let model;
IMG_SIZE = 150;

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

async function predict(img) {
  console.log("predicting...");
  console.log(`img: ${img}`);

//   create tensor
  let imgTensor = tf.browser.fromPixels(img);

  //   convert to grayscale
  imgTensor = tf.image.rgbToGrayscale(imgTensor);

//   resize and normalize
  imgTensor = tf.image.resizeBilinear(imgTensor, [IMG_SIZE, IMG_SIZE]);
  imgTensor = imgTensor.div(tf.scalar(255));
  imgTensor = imgTensor.expandDims();
  singleImagePlot(imgTensor);
  let prediction = model.predict(imgTensor);

  // Log the prediction
  console.log(prediction);
  // Convert the prediction to a binary value
 let binaryPrediction = prediction.argmax(1)
 console.log(binaryPrediction)
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
    image.onload = function () {
      console.log(image.width, image.height);
      predict(image);
    };
    image.src = dataURL;
  });
  // Read the file content as a data URL
  reader.readAsDataURL(file);
}
var form = document.getElementById("diagnosis-form");
form.addEventListener("submit", submitForm);

async function singleImagePlot(image) {
  const canvas = document.createElement("canvas");
  canvas.width = 28;
  canvas.height = 28;
  canvas.style = "margin: 4px;";
  await tf.browser.toPixels(image, canvas);
  return canvas;
}
