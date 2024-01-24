import * as tf from '@tensorflow/tfjs';
import { loadGraphModel, loadLayersModel } from '@tensorflow/tfjs';

MODEL_URL = './saved_models/model23/tfjs/model23/model.json'
async function loadModel(MODEL_URL){
    const model = await tf.loadGraphModel(MODEL_URL);
    console.log("Model loaded successfully");
    console.log(model);
    return model;
}

model = await loadGraphModel(MODEL_URL);
console.log('model loaded successfully');



