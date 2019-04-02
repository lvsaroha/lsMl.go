# lsMl.go
lsMl is a GO package for machine learning.
###### By love saroha (lvsaroha@gmail.com)

---------------------------------------

### Features
  * Lightweight and Fast
  * Native Go implementation. No C-bindings, just pure Go
  * Support for tensor operations like (Multiply, Divide, Add, Subtract, Square ...)
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

Tensor function to create a tensor object
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

