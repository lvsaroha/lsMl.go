package lsMl

import (
//"fmt"
)

// Model structure
type ModelObject struct {
	Layers       []LayerObject
	Loss         func(predict TensorObject, output TensorObject, batchSize TensorObject) TensorObject
	Optimizer    func(m *ModelObject, batchOutput TensorObject, batchSize TensorObject, inputArg TensorObject, output TensorObject)
	LearningRate TensorObject
}

// Model configuration
type ModelConfig struct {
	Loss         string
	Optimizer    string
	LearningRate float64
}

// Layer structure
type LayerObject struct {
	Weights         TensorObject
	Dweights        TensorObject
	Dbiases         TensorObject
	Biases          TensorObject
	Output          TensorObject
	WeightedSum     TensorObject
	Units           int
	ActivationFunc  func(value float64) float64
	DactivationFunc func(value float64) float64
	InputSize       int
	Err             TensorObject
}

// Layer configuration
type LayerConfig struct {
	Units      int
	Activation string
	InputShape []int
}

// Add layer
func (m *ModelObject) AddLayer(config LayerConfig) {
	var inputSize = 1
	var layer LayerObject
	// Set default
	if len(m.Layers) == 0 {
		// First layer config
		if len(config.InputShape) == 0 {
			// No units or input shape defined
			return
		}
		for i := 0; i < len(config.InputShape); i++ {
			inputSize *= config.InputShape[i]
		}
		if config.Units == 0 {
			// First layer
			config.Units = inputSize
		}
	} else {
		// Not first layer
		if config.Units == 0 {
			config.Units = m.Layers[len(m.Layers)-1].Units
		}
		inputSize = m.Layers[len(m.Layers)-1].Units
	}
	switch config.Activation {
	// Set activation functions
	case "sigmoid":
		layer.ActivationFunc = Sigmoid
		layer.DactivationFunc = Dsigmoid
		break
	case "relu":
		layer.ActivationFunc = Relu
		layer.DactivationFunc = Drelu
		break
	default:
		layer.ActivationFunc = Sigmoid
		layer.DactivationFunc = Dsigmoid
	}
	layer.Units = config.Units
	layer.InputSize = inputSize
	m.Layers = append(m.Layers, layer)
}

// Make model
func (m *ModelObject) Make(config ModelConfig) {
	// Set loss function
	switch config.Loss {
	case "meanSquareError":
		m.Loss = meanSquareError
		break
	default:
		m.Loss = meanSquareError
	}
	// Set optimizer algo
	switch config.Optimizer {
	case "sgd":
		m.Optimizer = gradientDescent
		break
	default:
		m.Optimizer = gradientDescent
	}
	if config.LearningRate == 0.0 {
		m.LearningRate = Tensor([]int{}, 0.2)
	} else {
		m.LearningRate = Tensor([]int{}, config.LearningRate)
	}
	// Create weights and biases in layers
	for i := 0; i < len(m.Layers); i++ {
		m.Layers[i].Weights = Tensor([]int{m.Layers[i].Units, m.Layers[i].InputSize}).Random()
		m.Layers[i].Biases = Tensor([]int{m.Layers[i].Units}).Random()
	}
}

// Predict
func (m ModelObject) Predict(input TensorObject) TensorObject {
	var i = 0
	// Feedforward
	for i = 0; i < len(m.Layers); i++ {
		m.Layers[i].WeightedSum = m.Layers[i].Weights.Mul(input).Add(m.Layers[i].Biases)
		m.Layers[i].Output = m.Layers[i].WeightedSum.Map(m.Layers[i].ActivationFunc)
		input = m.Layers[i].Output
	}
	return m.Layers[i-1].Output // Last layer
}

// COnfiguration for training
type TrainConfig struct {
	Epochs    int
	BatchSize int
	Shuffle   bool
	EachEpoch func(loss float64, epochNo int, batchNo int)
}

// Train
func (m *ModelObject) Train(inputs TensorObject, outputs TensorObject, config TrainConfig) {
	// Set default config
	if config.BatchSize == 0 || config.BatchSize < 1 {
		config.BatchSize = 1 // SGD
	}
	if config.Epochs == 0 || config.Epochs < 1 {
		config.Epochs = 100
	}
	inputs = inputs.Transpose()
	outputs = outputs.Transpose()
	// Make batches
	inputBatches := inputs.MakeBatches(config.BatchSize)
	outputBatches := outputs.MakeBatches(config.BatchSize)
	// For each epochs
	for i := 0; i < config.Epochs; i++ {
		// If shuffle
		if config.Shuffle == true {
			shuffle(&inputBatches, &outputBatches)
		}
		// For each batch
		for b := 0; b < len(inputBatches); b++ {
			// Take a batch and predict
			batchOutput := m.Predict(inputBatches[b])
			batchSize := Tensor([]int{}, inputBatches[b].Shape[1])
			if config.EachEpoch != nil {
				config.EachEpoch(m.Loss(batchOutput, outputBatches[b], batchSize).AddAll(), i, b)
			}
			// Run optimizer algorithm
			m.Optimizer(m, batchOutput, batchSize, inputBatches[b], outputBatches[b])
		}
	}
}

// Exported function
func Model() ModelObject {
	return ModelObject{}
}

// Suffle
func shuffle(arg1 *[]TensorObject, arg2 *[]TensorObject) {
	for i := 0; i < len(*arg1); i++ {
		p1 := int(Random(0.0, len(*arg1)))
		p2 := int(Random(0.0, len(*arg1)))
		temp1 := (*arg1)[p1]
		temp2 := (*arg2)[p1]
		(*arg1)[p1] = (*arg1)[p2]
		(*arg2)[p1] = (*arg2)[p2]
		(*arg1)[p2] = temp1
		(*arg2)[p2] = temp2
	}
}

// MSE loss function
func meanSquareError(predict TensorObject, Output TensorObject, batchSize TensorObject) TensorObject {
	var rts TensorObject = TensorObject{Shape: []int{}, Values: 0.0}
	rts.Values = predict.Sub(Output).Square().AddAll() / batchSize.Values.(float64)
	return rts
}

// Gradient descent optimizer
func gradientDescent(m *ModelObject, batchOutput TensorObject, batchSize TensorObject, inputArg TensorObject, output TensorObject) {
	m.Layers[len(m.Layers)-1].Err = batchOutput.Sub(output)
	// error . d(z)
	errD := m.Layers[len(m.Layers)-1].Err.Mul(m.Layers[len(m.Layers)-1].Output.Map(m.Layers[len(m.Layers)-1].DactivationFunc)).Div(batchSize)
	m.Layers[len(m.Layers)-1].Dbiases = errD.AddCols()
	var input TensorObject = inputArg
	// error . d(z) * T(I)
	if len(m.Layers) > 1 {
		input = m.Layers[len(m.Layers)-2].Output
	}
	m.Layers[len(m.Layers)-1].Dweights = errD.Mul(input.Transpose())
	m.Layers[len(m.Layers)-1].Weights = m.Layers[len(m.Layers)-1].Weights.Sub(m.Layers[len(m.Layers)-1].Dweights.Mul(m.LearningRate))
	m.Layers[len(m.Layers)-1].Biases = m.Layers[len(m.Layers)-1].Biases.Sub(m.Layers[len(m.Layers)-1].Dbiases.Mul(m.LearningRate))
	// Back propagation
	for j := len(m.Layers) - 2; j >= 0; j-- {
		m.Layers[j].Err = m.Layers[j+1].Weights.Transpose().Mul(m.Layers[j+1].Err).Mul(m.Layers[j].Output.Map(m.Layers[j].DactivationFunc)).Div(batchSize)
		// error * T(I)
		if j == 0 {
			input = inputArg
		} else {
			input = m.Layers[j-1].Output
		}
		m.Layers[j].Dbiases = m.Layers[j].Err.AddCols()
		m.Layers[j].Dweights = m.Layers[j].Err.Mul(input.Transpose())
		m.Layers[j].Weights = m.Layers[j].Weights.Sub(m.Layers[j].Dweights.Mul(m.LearningRate))
		m.Layers[j].Biases = m.Layers[j].Biases.Sub(m.Layers[j].Dbiases.Mul(m.LearningRate))
	}
}
