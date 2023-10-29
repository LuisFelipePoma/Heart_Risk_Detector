// PATH TO MODELS
const PATH = '../../models/'

// PARAMS NAMES FROM URL
const PARAMS_NAMES = [
  'age',
  'sex',
  'chestPain',
  'heartRate',
  'exercise',
  'thalassemia'
]

// LABELS FOR PREDICTION
const POSITIVE_PREDICT =
  '<span class="blueSpan">It seems there is a </span><span class="redSpan">risk of heart attack.</span>'

const NEGATIVE_PREDICT =
  '<span class="blueSpan">It seems there is </span><span class="redSpan">no risk of heart attack.</span>'

// MODELS PREDICTS
const predictsModels = {
  Perceptron: null,
  'Naive Bayes': null,
  CNN: null
}

// FUNCTION TO SHOW THE PREDICTION
const showPredict = e => {
  // Remove active class
  document.querySelectorAll('.alg').forEach(e => {
    e.classList.remove('active')
  })
  // Add active class
  e.classList.add('active')
  // Get name model
  const nameModel = e.innerHTML
  const prediction = predictsModels[nameModel]
  // Show Results
  document.getElementById('results').innerHTML = prediction
    ? POSITIVE_PREDICT
    : NEGATIVE_PREDICT
}

// FUNCTION TO SHOW THE PREDICTION
async function init () {
  // Get Params from URL
  let params = new URLSearchParams(location.search)
  // Get Data from URL
  let inputData = PARAMS_NAMES.map(name => {
    const value = params.get(name)
    return value ? parseFloat(value) : 0
  })
  // Load Models and Predict
  await loadModels(inputData)
}
async function loadModels (inputData) {
  try {
    // Load Models
    const modelGNB = await ort.InferenceSession.create(PATH + 'model_gnb.onnx')
    const modelPerceptron = await tf.loadLayersModel(
      '../../models/perceptron_model/model.json'
    )
    const modelCNN = await tf.loadLayersModel(
      '../../models/cnn_model/model.json'
    )

    // Transform data to predictGNB
    const data = Float32Array.from(inputData)
    const tensorSklearn = new ort.Tensor('float32', data, [1, inputData.length])
    const tensorKeras = tf.tensor2d([inputData])

    // Predict Results
    // ----- Gaussian Naive Bayes -----
    const results = await modelGNB.run({ input: tensorSklearn }, ['output_label'])
    const predictGNB = results.output_label.data

    // ----- Perceptron -----
    const predictPerceptron = await modelPerceptron.predict(tensorKeras).dataSync()
    const resultPerceptron = predictPerceptron > 0.5 ? 1 : 0

    // ----- Convolutional Neural Network -----
    const predictCNN = await modelCNN.predict(tensorKeras).dataSync()
    const resultCNN = predictCNN > 0.5 ? 1 : 0

    // Save Results
    predictsModels['Naive Bayes'] = Number(predictGNB[0])
    predictsModels['Perceptron'] = resultPerceptron
    predictsModels['CNN'] = resultCNN
  } catch (e) {
    console.error(`failed to load models: ${e}.`)
  }
}