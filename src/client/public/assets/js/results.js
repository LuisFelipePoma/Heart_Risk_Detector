// GLOBAL VARIABLES
const PATH = '../../models/'
const PARAMS_NAMES = [
  'age',
  'sex',
  'chestPain',
  'heartRate',
  'exercise',
  'thalassemia'
]

const predictsModels = {
  Perceptron: null,
  GaussianNaiveBayes: null,
  CNN: null
}

// On load function form html
async function init () {
  console.log('Cargando modelo...')

  let params = new URLSearchParams(location.search)

  let inputData = PARAMS_NAMES.map(name => {
    const value = params.get(name)
    return value ? parseFloat(value) : 0
  })
  await loadModels(inputData)
  console.log('Modelo cargado...')
}

async function loadModels (inputData) {
  // Load the ONNX model
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

    // Feed inputs and run
    const feeds = { input: tensorSklearn }
    const results = await modelGNB.run(feeds, ['output_label'])

    // Predict Results
    const predictGNB = results.output_label.data
    const predictPerceptron = modelPerceptron.predict(tensorKeras).dataSync()
    const resultPerceptron = predictPerceptron > 0.5 ? 1 : 0
    const predictCNN = modelCNN.predict(tensorKeras).dataSync()
    const resultCNN = predictCNN > 0.5 ? 1 : 0

    // Save Results
    predictsModels['GaussianNaiveBayes'] = Number(predictGNB[0])
    predictsModels['Perceptron'] = resultPerceptron
    predictsModels['CNN'] = resultCNN
    console.log(predictsModels)
  } catch (e) {
    console.error(`failed to load models: ${e}.`)
  }
}
