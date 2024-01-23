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
  const model = await tf.loadLayersModel(MODEL_URL);
  console.log("Model loaded successfully");
  console.log(model.summary());
  return model;
}

async function predict(x){
    console.log(x);
}
MODEL_URL = "saved_models/tfjs/model23/model.json";
model = loadModel(MODEL_URL);


