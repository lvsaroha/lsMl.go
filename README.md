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

Random function to generate random values between given range
`lsMl.Tensor([]int).Random( min? , max? , floor? )`<br />
*Supported types for all arguments (float64)*

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
