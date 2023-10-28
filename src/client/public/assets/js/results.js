// On load function form html
const PATH = '../../models/'
const PARAMS_NAMES = [
  'age',
  'sex',
  'chestPain',
  'heartRate',
  'exercise',
  'thalassemia'
]

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
    const session = await ort.InferenceSession.create(PATH + 'model_gnb.onnx')
    // const out = session.get_outputs()
    const data = Float32Array.from(inputData)
    const tensor = new ort.Tensor('float32', data, [1, inputData.length])
    // Define the correct input name based on your model's expectations

    // Feed inputs and run
    const feeds = { input: tensor }

    const results = await session.run(feeds, ['output_label'])

    const predict = results.output_label.data
		console.log(predict[0])
  } catch (e) {
    console.error(`failed to inference ONNX model: ${e}.`)
  }
}

