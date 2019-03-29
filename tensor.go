package lsMl

// A machine learning package
// By love saroha (lvsaroha@gmail.com)
// GNU General Public License v3.0

// Imported packages
import (
	"fmt"
	"math"
)

// Tensor structure
type TensorObject struct {
	Shape  []int
	Values interface{}
}

// Print tensor
func (ts TensorObject) Print() {
	switch v := ts.Values.(type) {
	case float64:
		// Scalar
		fmt.Printf("\n %f \n %s \n\n", v, "Scalar")
		break
	case [][]float64:
		// Matrix
		fmt.Printf("\n")
		for i := 0; i < ts.Shape[0]; i++ {
			fmt.Printf(" [ ")
			for j := 0; j < ts.Shape[1]; j++ {
				fmt.Printf(" %f ", v[i][j])
			}
			fmt.Printf(" ]  \n")
		}
		fmt.Printf(" %s %v %s %v %s \n\n", " Matrix : (", ts.Shape[0], "x", ts.Shape[1], ")")
	}
}

// Copy tensor
func (ts TensorObject) Copy() TensorObject {
	var rts = TensorObject{}
	// Copy shape
	for i := 0; i < len(ts.Shape); i++ {
		rts.Shape = append(rts.Shape, ts.Shape[i])
	}
	// Check shape
	switch len(ts.Shape) {
	case 0: // If scalar
		rts.Values = ts.Values
		break
	case 2: // Matrix
		rts.Values = [][]float64{}
		matrix := ts.Values.([][]float64)
		for i := 0; i < ts.Shape[0]; i++ {
			r := make([]float64, ts.Shape[1])
			for j := 0; j < ts.Shape[1]; j++ {
				r[j] = matrix[i][j]
			}
			rts.Values = append(rts.Values.([][]float64), r)
		}
	}
	return rts
}

// Random tensor values
func (ts TensorObject) Random(args ...interface{}) TensorObject {
	var rts = ts.Copy()
	// Set default
	var min = -1.0
	var max = 1.0
	var floor = false
	// Assign arguments
	for k, arg := range args {
		if k == 0 {
			// Min argument value
			switch v := arg.(type) {
			case int:
				min = float64(v)
			}
		} else if k == 1 {
			// Max argument
			switch v := arg.(type) {
			case int:
				max = float64(v)
			}
		} else if k == 2 {
			switch v := arg.(type) {
			case bool:
				floor = v
			}
		}
	}
	// Check shape
	switch len(ts.Shape) {
	case 0: // If scalar
		rts.Values = Random(min, max, floor)
		break
	case 2: // Matrix
		matrix := rts.Values.([][]float64)
		for i := 0; i < rts.Shape[0]; i++ {
			for j := 0; j < rts.Shape[1]; j++ {
				matrix[i][j] = RandomWith(min, max, floor, float64(i)*math.Pow(10.0, float64(rts.Shape[0]))+float64(j))
			}
		}
		rts.Values = matrix
	}
	return rts
}

// Add tensor
func (ts TensorObject) Add(arg TensorObject) TensorObject {
	return elementWise(ts, arg, 0)
}

// Subtract tensor
func (ts TensorObject) Sub(arg TensorObject) TensorObject {
	return elementWise(ts, arg, 1)
}

// Multiplication
func (ts TensorObject) Mul(arg TensorObject) TensorObject {
	return elementWise(ts, arg, 2)
}

// Square
func (ts TensorObject) Square() TensorObject {
	return elementWise(ts, ts, 2)
}

// Divide
func (ts TensorObject) Div(arg TensorObject) TensorObject {
	return elementWise(ts, arg, 3)
}

// Transpose
func (ts TensorObject) Transpose() TensorObject {
	var rts TensorObject
	// type
	switch len(ts.Shape) {
	case 1:
		rts.Values = ts.Values
	case 2:
		rts.Shape = []int{ts.Shape[1], ts.Shape[0]}
		values := make([][]float64, rts.Shape[0])
		matrix := ts.Values.([][]float64)
		// Matrix or vector
		for i := 0; i < rts.Shape[0]; i++ {
			r := make([]float64, rts.Shape[1])
			for j := 0; j < rts.Shape[1]; j++ {
				r[j] = matrix[j][i]
			}
			values[i] = r
		}
		rts.Values = values
	}
	return rts
}

// Map function
func (ts TensorObject) Map(callback func(value float64) float64) TensorObject {
	var rts = ts.Copy()
	switch len(ts.Shape) {
	case 0: // Scalar
		value := ts.Values.(float64)
		value = callback(value)
		rts.Values = value
		break
	case 2: // Matrix or vector
		values := ts.Values.([][]float64)
		for i := 0; i < rts.Shape[0]; i++ {
			for j := 0; j < rts.Shape[1]; j++ {
				values[i][j] = callback(values[i][j])
			}
		}
		rts.Values = values
	}
	return rts
}

// Add all
func (ts TensorObject) AddAll() float64 {
	var sum float64
	switch len(ts.Shape) {
	case 0: // Scalar
		return ts.Values.(float64)
	case 2: // Matrix
		values := ts.Values.([][]float64)
		for i := 0; i < ts.Shape[0]; i++ {
			for j := 0; j < ts.Shape[1]; j++ {
				sum += values[i][j]
			}
		}
		return sum
	}
	return sum
}

// Values of a matrix tensor
func (ts TensorObject) Value() [][]float64 {
	switch len(ts.Shape) {
	case 1:
		return [][]float64{[]float64{ts.Values.(float64)}}
	case 2:
		return ts.Values.([][]float64)
	}
	return [][]float64{}
}

// Col extend
func (ts TensorObject) ColExtend(scale int) TensorObject {
	var rts TensorObject
	// Check shape
	switch len(ts.Shape) {
	case 0: //Scalar
		return ts
		break
	case 2: // Matrix
		rts.Shape = []int{ts.Shape[0], ts.Shape[1] * scale}
		matrix := make([][]float64, ts.Shape[0])
		values := ts.Values.([][]float64)
		for i := 0; i < ts.Shape[0]; i++ {
			r := make([]float64, rts.Shape[1])
			for j := 0; j < rts.Shape[1]; j++ {
				r[j] = values[i][j/scale]
			}
			matrix[i] = r
		}
		rts.Values = matrix
		break
	}
	return rts
}

// Add columns
func (ts TensorObject) AddCols() TensorObject {
	var rts TensorObject
	if len(ts.Shape) == 0 {
		// Scalar
		return ts
	} else if len(ts.Shape) == 2 {
		rts.Shape = []int{ts.Shape[0], 1}
		values := make([][]float64, rts.Shape[0])
		matrix := ts.Values.([][]float64)
		for i := 0; i < ts.Shape[0]; i++ {
			sum := 0.0
			for j := 0; j < ts.Shape[1]; j++ {
				sum += matrix[i][j]
			}
			values[i] = []float64{sum}
		}
		rts.Values = values
	}
	return rts
}

// Make batches
func (ts TensorObject) MakeBatches(size int) []TensorObject {
	var rts []TensorObject
	if len(ts.Shape) == 0 {
		// Scalar
		return []TensorObject{ts.Copy()}
	} else if len(ts.Shape) == 2 {
		// Matrix
		totalBatches := ts.Shape[1] / size
		if totalBatches*size != ts.Shape[1] {
			totalBatches += 1
		}
		matrix := ts.Values.([][]float64)
		initial := size
		for t := 0; t < totalBatches; t++ {
			initial = t * size
			limit := initial + size
			if limit > ts.Shape[1] { // More than columns
				limit = ts.Shape[1]
			}
			var nts TensorObject = TensorObject{Shape: []int{ts.Shape[0], limit - initial}}
			values := [][]float64{}
			for i := 0; i < ts.Shape[0]; i++ {
				r := make([]float64, limit-initial)
				var c = 0
				for j := initial; j < limit; j++ {
					r[c] = matrix[i][j]
					c++
				}
				values = append(values, r)
			}
			nts.Values = values
			rts = append(rts, nts)
		}
	} else {
		return []TensorObject{ts}
	}
	return rts
}

// Export function
func Tensor(args ...interface{}) TensorObject {
	// Default values
	var shape []int = []int{}
	var values interface{}
	var create = false

	// Assign arguments
	for k, arg := range args {
		if k == 0 {
			// Shape argument value
			shape = arg.([]int)
		} else if k == 1 {
			// Values argument
			values = arg
		}
	}
	// Create value copy
	var valuesFinal = values

	// If tensor shape
	if len(shape) == 0 {
		// Scalar
		switch v := values.(type) {
		case int:
			valuesFinal = float64(v)
		case float64: // Do nothing
			break
		default:
			create = true
		}
	} else if len(shape) == 1 {
		// Vector
		switch v := values.(type) {
		case []int:
			if len(v) != shape[0] {
				// Invalid size
				return TensorObject{Shape: []int{}, Values: 0.0}
			}
			valuesFinal = toMatrix(sliceToF64(v))
			shape = []int{shape[0], 1}
			break
		case []float64: // Do nothing
			if len(v) != shape[0] {
				// Invalid size
				return TensorObject{Shape: []int{}, Values: 0.0}
			}
			shape = []int{shape[0], 1}
			valuesFinal = toMatrix(v)
			break
		default:
			create = true
		}
	} else if len(shape) == 2 {
		// Matrix
		switch v := values.(type) {
		case [][]int:
			if len(v) != shape[0] {
				// Invalid size
				return TensorObject{Shape: []int{}, Values: 0.0}
			}
			valuesFinal = slice2dToF64(v)
			break
		case [][]float64: // Do nothing
			if len(v) != shape[0] {
				// Invalid size
				return TensorObject{Shape: []int{}, Values: 0.0}
			}
			break
		default:
			create = true
		}
	} else {
		// Not supported
		return TensorObject{Shape: []int{}, Values: 0.0}
	}
	if create == true {
		// Create values from shape
		if len(shape) == 0 {
			// Scalar
			valuesFinal = 0.0
		} else if len(shape) == 1 {
			// Vector
			vector := [][]float64{}
			for i := 0; i < shape[0]; i++ {
				vector = append(vector, []float64{0.0})
			}
			shape = []int{shape[0], 1}
			valuesFinal = vector
		} else if len(shape) == 2 {
			// Matrix
			matrix := [][]float64{}
			for i := 0; i < shape[0]; i++ {
				matrix = append(matrix, []float64{})
				for j := 0; j < shape[1]; j++ {
					matrix[i] = append(matrix[i], 0.0)
				}
			}
			valuesFinal = matrix
		}
	}
	return TensorObject{Shape: shape, Values: valuesFinal}
}

// Element wise operation
func elementWise(ts TensorObject, arg TensorObject, opt int) TensorObject {
	// Create result tensor
	var rts TensorObject = ts.Copy()
	switch len(ts.Shape) {
	case 0: // If scalar
		// Check arg shape
		switch len(arg.Shape) {
		case 0: // Scalar
			switch opt {
			case 0: // Addition
				rts.Values = rts.Values.(float64) + arg.Values.(float64)
				break
			case 1: // Subtraction
				rts.Values = rts.Values.(float64) - arg.Values.(float64)
				break
			case 2: // Multiplication
				rts.Values = rts.Values.(float64) * arg.Values.(float64)
				break
			case 3: // Division
				rts.Values = rts.Values.(float64) / arg.Values.(float64)
				break
			}
			break
		case 2: // Matrix
			rts.Shape = arg.Shape
			matrix := arg.Values.([][]float64)
			for i := 0; i < arg.Shape[0]; i++ {
				for j := 0; j < arg.Shape[1]; j++ {
					switch opt {
					case 0: // Addition
						matrix[i][j] = rts.Values.(float64) + matrix[i][j]
						break
					case 1: // Subtraction
						matrix[i][j] = rts.Values.(float64) - matrix[i][j]
						break
					case 2: // Multiplication
						matrix[i][j] = rts.Values.(float64) * matrix[i][j]
						break
					case 3: // Division
						matrix[i][j] = rts.Values.(float64) / matrix[i][j]
						break
					}
				}
			}
			rts.Values = matrix
		}
		break
	case 2: // if matrix
		matrix := rts.Values.([][]float64)
		// Check arg shape
		switch len(arg.Shape) {
		case 0: // Scalar
			for i := 0; i < rts.Shape[0]; i++ {
				for j := 0; j < rts.Shape[1]; j++ {
					switch opt {
					case 0: // Addition
						matrix[i][j] = matrix[i][j] + arg.Values.(float64)
						break
					case 1: // Subtraction
						matrix[i][j] = matrix[i][j] - arg.Values.(float64)
						break
					case 2: // Multiplication
						matrix[i][j] = matrix[i][j] * arg.Values.(float64)
						break
					case 3: // Division
						matrix[i][j] = matrix[i][j] / arg.Values.(float64)
					}
				}
			}
			break
		case 2: // Matrix
			matrixArg := arg.Values.([][]float64)
			// Check both dimensions
			if ts.Shape[1] == arg.Shape[0] && opt == 2 { // Multiply matrix
				rts.Shape = []int{ts.Shape[0], arg.Shape[1]}
				values := make([][]float64, ts.Shape[0])
				for i := 0; i < rts.Shape[0]; i++ {
					r := make([]float64, rts.Shape[1])
					for j := 0; j < rts.Shape[1]; j++ {
						var sum float64
						for k := 0; k < ts.Shape[1]; k++ {
							sum += matrix[i][k] * matrixArg[k][j]
						}
						r[j] = sum
					}
					values[i] = r
				}
				rts.Values = values
				return rts
			}
			if ts.Shape[0] != arg.Shape[0] || arg.Shape[1] != ts.Shape[1] && opt == 2 { // Hadamard product
				if ts.Shape[0] == arg.Shape[0] && arg.Shape[1] == 1 {
					arg = arg.ColExtend(ts.Shape[1])
				} else {
					return rts
				}
			}
			for i := 0; i < arg.Shape[0]; i++ {
				for j := 0; j < arg.Shape[1]; j++ {
					switch opt {
					case 0: // Addition
						matrix[i][j] = matrix[i][j] + matrixArg[i][j]
						break
					case 1: // Subtraction
						matrix[i][j] = matrix[i][j] - matrixArg[i][j]
						break
					case 2: // Multiplication
						matrix[i][j] = matrix[i][j] * matrixArg[i][j]
						break
					case 3: // Division
						matrix[i][j] = matrix[i][j] / matrixArg[i][j]
					}
				}
			}
		}
		rts.Values = matrix
	}
	return rts
}

// Convert [] to []float64
func sliceToF64(val interface{}) []float64 {
	var rs []float64
	switch v := val.(type) {
	case []int:
		for i := 0; i < len(v); i++ {
			rs = append(rs, float64(v[i]))
		}
		break
	case []int64:
		for i := 0; i < len(v); i++ {
			rs = append(rs, float64(v[i]))
		}
		break
	case []float32:
		for i := 0; i < len(v); i++ {
			rs = append(rs, float64(v[i]))
		}
	}
	return rs
}

// Convert [][] to [][]float64
func slice2dToF64(val interface{}) [][]float64 {
	var rs [][]float64
	switch v := val.(type) {
	case [][]int:
		for i := 0; i < len(v); i++ {
			rs = append(rs, sliceToF64(v[i]))
		}
		break
	case [][]int64:
		for i := 0; i < len(v); i++ {
			rs = append(rs, sliceToF64(v[i]))
		}
		break
	case [][]float32:
		for i := 0; i < len(v); i++ {
			rs = append(rs, sliceToF64(v[i]))
		}
	}
	return rs
}

// Vector to matrix
func toMatrix(value []float64) [][]float64 {
	// Matrix
	matrix := [][]float64{}
	for i := 0; i < len(value); i++ {
		matrix = append(matrix, []float64{})
		for j := 0; j < 1; j++ {
			matrix[i] = append(matrix[i], value[i])
		}
	}
	return matrix
}
