# lsMl.go
lsMl is a GO package for machine learning.
###### By love saroha (lvsaroha@gmail.com)

---------------------------------------

### Features
  * Lightweight and Fast
  * Native Go implementation. No C-bindings, just pure Go
  * Support for [tensor](#Creating-a-tensor-object-in-lsMl.go) operations like (Multiply, Divide, Add, Subtract, Square ...)
  * Supports sequential models 
  * Supports loss functions like (Mean Square Error)
  * Supports opitmizer algorithms like (Gradient Descent)

### Requirements
  * Go 1.9 or higher. We aim to support the 3 latest versions of Go.

---------------------------------------

### Installation
Simple install the package to your [$GOPATH](https://github.com/golang/go/wiki/GOPATH "GOPATH") with the [go tool](https://golang.org/cmd/go/ "go command") from shell:
```bash
$ go get -u github.com/lvsaroha/lsMl.go
```
Make sure [Git is installed](https://git-scm.com/downloads) on your machine and in your system's `PATH`.

---------------------------------------
### Creating a tensor object in lsMl.go

**Tensor()** function to create a tensor object
`lsMl.Tensor( shape , values?)`<br />
*Supported type for shape is []int and for values ([]int , []float64 , [][]int , [][]float64)*

```Golang    
    lsMl.Tensor([]int)
    // Returns a scalar tensor with value 0.0 

    lsMl.Tensor([]int , 2.3)
    // Returns a scalar tensor with value 2.3     

    lsMl.Tensor([3]int)
    // Returns a vector of size 3 with values 0.0

    lsMl.Tensor([3]int , []int{3, 4, 7})
    // Returns a vector of size 3 with value 3.0, 4.0, 7.0

    lsMl.Tensor([3,2]int)
    // Returns a matrix of shape 3x2 with all values 0.0

    lsMl.Tensor([3,2]int , [][]int{[]int{2, 3} , []int{4, 5} , []int{6, 7}})
    // Returns a matrix of shape 3x2 with all values [[2, 3], [4, 5], [6, 7]]    
```

##### Random
**Random()** function to generate random values between given range
`lsMl.Tensor([]int).Random( min? , max? , floor? )`<br />
*Supported types for all arguments (float64)*
> Default values for minimum , maximum and floor are (-1, 1, false)

```Golang    
    v := lsMl.Tensor([3]int)
    // Returns a vector of size 3 with values 0.0

    v2 := v.Random()
    // Returns a vector of size 3 with values between -1 and 1 

    v3 := lsMl.Tensor([4]int).Random()
    // Returns a vector of size 4 with values between -1 and 1 

    m := lsMl.Tensor([3,3]int).Random(0, 10)
    // Returns a matrix of shape 3x3 with values between 0 and 10 (Example 0.3 , 4.5 , 2.3 , 7.8 , 9.3 ...)

    m := lsMl.Tensor([3,3]int).Random(0, 10, true)
    // Returns a matrix of shape 3x3 with non-decimal values between 0 and 10 (Example 2.0, 1.0, 4.0, 7.0, 8.0 ...)    
```
---------------------------------------

### Tensor operations

##### Addition
**Add()** function to add two tensor objects
`TensorObject.Add(TensorObject)`<br />
*Supported types for argument (TensorObject)*
<br>
>For addition of two tensor object , if none of them is a scalar than the shape of the both tensors must be same for element wise addition.

```Golang    
    v := lsMl.Tensor([]int)
    // Returns a scalar tensor of value 0

    v2 := v.Random()
    // Returns a scalar tensor with value between -1 and 1 

    v3 := lsMl.Tensor([4]int).Random()
    // Returns a vector of size 4 with values between -1 and 1 

    s := v2.Add(v3)
    // Returns a tensor with added values of v2 and v3 and shape of v3

    m := lsMl.Tensor([3,2]int).Random(0, 10, true)
    // Returns a matrix of shape 3x2 with non-decimal values between 0 and 10 (Example 2.0, 1.0, 4.0, 7.0, 8.0 ...)   

    m2 := lsMl.Tensor([3,2]int).Random(0, 10, true)
    // Returns a matrix of shape 3x2 with non-decimal values between 0 and 10 (Example 2.0, 1.0, 4.0, 7.0, 8.0 ...)  

    s2 := m.Add(m2)
    // Returns a matrix of shape 3x2 with added values of m and m2        
```

##### Subtraction
**Sub()** function to subtract value of one tensor from another
`TensorObject.Sub(TensorObject)`<br />
*Supported types for argument (TensorObject)*
<br>
>For subtraction of two tensor object , if none of them is a scalar than the shape of the both tensors must be same for element wise subtraction.

```Golang    
    v := lsMl.Tensor([]int)
    // Returns a scalar tensor of value 0

    v2 := v.Random()
    // Returns a scalar tensor with value between -1 and 1 

    v3 := lsMl.Tensor([4]int).Random()
    // Returns a vector of size 4 with values between -1 and 1 

    s := v2.Sub(v3)
    // Returns a tensor with value of v2 subtracted by values of v3 and shape of v3

    m := lsMl.Tensor([3,2]int).Random(0, 10, true)
    // Returns a matrix of shape 3x2 with non-decimal values between 0 and 10 (Example 2.0, 1.0, 4.0, 7.0, 8.0 ...)   

    m2 := lsMl.Tensor([3,2]int).Random(0, 10, true)
    // Returns a matrix of shape 3x2 with non-decimal values between 0 and 10 (Example 2.0, 1.0, 4.0, 7.0, 8.0 ...)  

    s2 := m.Add(m2)
    // Returns a matrix of shape 3x2 with values of m subtracted by values of m2        
```

##### Multiply
**Mul()** function to multiply values of two tensor objects
`TensorObject.Mul(TensorObject)`<br />
*Supported types for argument (TensorObject)*
<br>
>For multiplication of two tensor object , if none of them is a scalar than the shape of the both tensors must be same for element wise multiplication (if both are matrix than hadamard product is calculated).<br>
If both tensor are matrix and their shapes are first(3x2) second(2x3) than the resultant matrix is (3x3) after multiplication.

```Golang    
    v := lsMl.Tensor([]int)
    // Returns a scalar tensor of value 0

    v2 := v.Random()
    // Returns a scalar tensor with value between -1 and 1 

    v3 := lsMl.Tensor([4]int).Random()
    // Returns a vector of size 4 with values between -1 and 1 

    s := v2.Mul(v3)
    // Returns a tensor with value of v2 multiply by values of v3 and shape of v3

    m := lsMl.Tensor([3,2]int).Random(0, 10, true)
    // Returns a matrix of shape 3x2 with non-decimal values between 0 and 10 (Example 2.0, 1.0, 4.0, 7.0, 8.0 ...)   

    m2 := lsMl.Tensor([3,2]int).Random(0, 10, true)
    // Returns a matrix of shape 3x2 with non-decimal values between 0 and 10 (Example 2.0, 1.0, 4.0, 7.0, 8.0 ...)  

    r2 := m.Mul(m2)
    // Returns a matrix of shape 3x2 with values of m multiply by values of m2        

    m3 := lsMl.Tensor([3,2]int).Random(0, 10, true)
    // Returns a matrix of shape 3x2 with non-decimal values between 0 and 10 (Example 2.0, 1.0, 4.0, 7.0, 8.0 ...)   

    m4 := lsMl.Tensor([2,3]int).Random(0, 10, true)
    // Returns a matrix of shape 3x2 with non-decimal values between 0 and 10 (Example 2.0, 1.0, 4.0, 7.0, 8.0 ...)  

    r3 := m3.Mul(m4)
    // Returns a matrix of shape 3x3 with values of m3 multiply by values of m4            
```

##### Division
**Div()** function to divide value of one tensor from another
`TensorObject.Div(TensorObject)`<br />
*Supported types for argument (TensorObject)*
<br>
>For division of two tensor object , if none of them is a scalar than the shape of the both tensors must be same for element wise division.

```Golang    
    v := lsMl.Tensor([]int)
    // Returns a scalar tensor of value 0

    v2 := v.Random()
    // Returns a scalar tensor with value between -1 and 1 

    v3 := lsMl.Tensor([4]int).Random()
    // Returns a vector of size 4 with values between -1 and 1 

    s := v2.Div(v3)
    // Returns a tensor with value of v2 divided by values of v3 and shape of v3

    m := lsMl.Tensor([3,2]int).Random(0, 10, true)
    // Returns a matrix of shape 3x2 with non-decimal values between 0 and 10 (Example 2.0, 1.0, 4.0, 7.0, 8.0 ...)   

    m2 := lsMl.Tensor([3,2]int).Random(0, 10, true)
    // Returns a matrix of shape 3x2 with non-decimal values between 0 and 10 (Example 2.0, 1.0, 4.0, 7.0, 8.0 ...)  

    s2 := m.Add(m2)
    // Returns a matrix of shape 3x2 with values of m divided by values of m2        
```

---------------------------------------

### Tensor functions

##### Print
**Print()** function prints values of the tensor in terminal
`TensorObject.Print()`<br />

```Golang
    m := lsMl.Tensor([3,2]int).Random(0, 10, true)
    // Returns a matrix of shape 3x2 with non-decimal values between 0 and 10 (Example 2.0, 1.0, 4.0, 7.0, 8.0 ...)   

    m.Print()
    // Shows values of m in terminal like [[2.0, 1.0], [4.0, 7.0], [8.0, 1.0]] matrix(3x2)
```

##### Transpose
**Transpose()** function rotate the tensor
`TensorObject.Transpose()`<br />
> If a tensor contains scalar value than transpose of the tensor returns same tensor

```Golang
    m := lsMl.Tensor([3,2]int).Random(0, 10, true)
    // Returns a matrix of shape 3x2 with non-decimal values between 0 and 10 (Example 2.0, 1.0, 4.0, 7.0, 8.0 ...)   

    r := m.Transpose()
    // Returns a matrix of shape(2x3)
```
##### Map
**Map()** function creates a tensor from a given tensor according to condition provided in callback
`TensorObject.Map(callback)`<br />
> Callback is a function which gets a value of the tensor as a argument and return a new value

```Golang
    m := lsMl.Tensor([3,2]int).Random(0, 10, true)
    // Returns a matrix of shape 3x2 with non-decimal values between 0 and 10 (Example 2.0, 1.0, 4.0, 7.0, 8.0 ...)   

    r := m.Map(func(val float64) float64 { return val * val } )
    // Returns matrix of shape (3x2) with values of m squared
```
---------------------------------------

##### Example of logistic regression using tensor operations
> In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model; it is a form of binomial regression. 

```Golang
    // Inputs , outputs and parameters 
    inputs := lsMl.Tensor([]int{2, 4}, [][]float64{[]float64{0, 0, 1, 1}, []float64{0, 1, 0, 1}})
    outputs := lsMl.Tensor([]int{1, 4}, [][]float64{[]float64{0, 1, 1, 1}}) // Desired output of OR GATE
    w := lsMl.Tensor([]int{1, 2}).Random() // Weight matrix of shape (1x2) with values between (-1 and 1)
    b := lsMl.Tensor([]int{}).Random() // Bias value between (-1 and 1)
    lr := lsMl.Tensor([]int{}, 0.3) // Learning rate
    bs := lsMl.Tensor([]int{} , 4) // Batch size

    for i := 0; i < 1000; i++ { // Applying gradient descent
      z := w.Mul(inputs).Add(b)
      a := z.Map(lsMl.Sigmoid) // Activation function sigmoid
      e := a.Sub(outputs).Div(bs)
      dw := e.Mul(inputs.Transpose())
      w = w.Sub(dw.Mul(lr))
      b = b.Sub(e.Mul(lr).AddCols())
    }
    // After training
    m := w.Mul(inputs).Add(b)
    n := m.Map(lsMl.Sigmoid)
    n.Print() // Print output
    // [  0.100241  0.954295  0.959951  0.998006  ]
```

---------------------------------------

### How to use models in lsMl

##### Creating a model object 

**Model()** function to create a model object
`lsMl.Model()`

```Golang
     m := lsMl.Model()
     // Returns a model object
```

##### Adding layers to the model object

**AddLayer()** function add layer to model object
`ModelObject.AddLayer(config)`<br>
> Config argument takes information about Units , Activation function and InputShape using LayerConfig Object. InputShape is required only for first layer config.
Default value for activation is sigmoid.  

```Golang
     m := lsMl.Model()
     // Returns a model object

     m.AddLayer(lsMl.LayerConfig{InputShape: []int{2} , Units: 3, Activation: "relu"})
     // Add layer to model m with units 3 and activation function relu

     m.AddLayer(lsMl.LayerConfig{Units: 1})
     // Add another layer to model m with units 2 and activation function sigmoid
```

##### Making the model for usage

**Make()** function makes the model with given configuration
`ModelObject.Make(config)`<br>
> Config argument takes information about Loss function , Optimizer  and LearningRate using ModelConfig Object.
Default values for Loss function , Optimizer and LearningRate are ("meanSquareError" , "sgd", 0.2). 

```Golang
     m := lsMl.Model()
     // Returns a model object

     m.AddLayer(lsMl.LayerConfig{InputShape: []int{2} , Units: 3, Activation: "relu"})
     // Add layer to model m with units 3 and activation function relu

     m.AddLayer(lsMl.LayerConfig{Units: 1})
     // Add another layer to model m with units 2 and activation function sigmoid

     m.Make(lsMl.ModelConfig{Loss: "meanSquareError", Optimizer: "sgd", LearningRate: 0.2})
     // Makes the model with given values of loss function , optimizer and learning rate
```

##### Training the model

**Train()** function to train the model with given configuration
`ModelObject.Train(inputs , outputs , config)`<br>
> Config argument takes information about Epochs , BatchSize  and Shuffle using TrainConfig Object.
Default values for Epochs , BatchSize and Shuffle are (100 , 1, false).

```Golang
     m := lsMl.Model()
     // Returns a model object

     m.AddLayer(lsMl.LayerConfig{InputShape: []int{2} , Units: 3, Activation: "relu"})
     // Add layer to model m with units 3 and activation function relu

     m.AddLayer(lsMl.LayerConfig{Units: 1})
     // Add another layer to model m with units 2 and activation function sigmoid

     m.Make(lsMl.ModelConfig{Loss: "meanSquareError", Optimizer: "sgd", LearningRate: 0.2})
     // Makes the model with given values of loss function , optimizer and learning rate

    inputs := lsMl.Tensor([]int{4, 2}, [][]float64{[]float64{0, 0}, []float64{1, 0}, []float64{0, 1}, []float64{1, 1}})
    outputs := lsMl.Tensor([]int{4}, []float64{0, 1, 1, 0}) // Desired output of XOR GATE

    m.Train(inputs, outputs, lsMl.TrainConfig{Epochs: 5000, BatchSize: 1, Shuffle: true})
    // Trains the model according to given configuration
```

##### Predicting with model

**Predict()** function predicts the output for a given input
`ModelObject.Predict(input)`<br>
> Input argument is a tensor object

```Golang
     m := lsMl.Model()
     // Returns a model object

     m.AddLayer(lsMl.LayerConfig{InputShape: []int{2} , Units: 3, Activation: "relu"})
     // Add layer to model m with units 3 and activation function relu

     m.AddLayer(lsMl.LayerConfig{Units: 1})
     // Add another layer to model m with units 2 and activation function sigmoid

     m.Make(lsMl.ModelConfig{Loss: "meanSquareError", Optimizer: "sgd", LearningRate: 0.2})
     // Makes the model with given values of loss function , optimizer and learning rate

    inputs := lsMl.Tensor([]int{4, 2}, [][]float64{[]float64{0, 0}, []float64{1, 0}, []float64{0, 1}, []float64{1, 1}})
    outputs := lsMl.Tensor([]int{4}, []float64{0, 1, 1, 0}) // Desired output of XOR GATE

    m.Train(inputs, outputs, lsMl.TrainConfig{Epochs: 5000, BatchSize: 1, Shuffle: true})
    // Trains the model according to given configuration

    m.Predict(lsMl.Tensor([]int{2}, []float64{0, 0})).Print() // Prints 0.011401 
    m.Predict(lsMl.Tensor([]int{2}, []float64{1, 0})).Print() // Prints 0.912271 
    m.Predict(lsMl.Tensor([]int{2}, []float64{0, 1})).Print() // Prints 0.939822 
    m.Predict(lsMl.Tensor([]int{2}, []float64{1, 1})).Print() // Prints 0.020241 
```
