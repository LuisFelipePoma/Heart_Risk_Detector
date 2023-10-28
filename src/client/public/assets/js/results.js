// On load function form html
const PATH = '../../models/'
// import * as onnx from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.min.js'
;(async () => {
  console.log('Cargando modelo...')
  // ml.load('../../models/modelo_gnb.pkl', function(model) {
  // 	// Hacer una predicción
  // 	var input_data = [/* tus datos de entrada */];
  // 	var prediction = model.predict(input_data);
  // 	console.log('Predicción:', prediction);
  // });
  loadModels()
  let params = new URLSearchParams(location.search)
  console.log(params.get('age'))
  console.log('Modelo cargado...')
})()

async function loadModels () {
  // Load the ONNX model
  try {
    const session = await ort.InferenceSession.create(PATH + 'model_gnb.onnx')
    const inputData = [42, 1, 1, 162, 0, 2] // Wrap your input data in an array
    const data = Float32Array.from(inputData)
    const tensorA = new ort.Tensor('float32', data)
    // feed inputs and run
    const feeds = { a: tensorA }
    const results = await session.run(feeds)

    const dataC = results.c.data
    document.write(`data of result tensor 'c': ${dataC}`)
  } catch (e) {
    alert(`failed to inference ONNX model: ${e}.`)
  }
}

function predict () {}
