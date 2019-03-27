package lsMl

// Imported packages
import (
	"math"
	"math/rand"
	"time"
)

// Random gives float64 number between range
func Random(args ...interface{}) float64 {
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
	rand.Seed(time.Now().UTC().UnixNano())
	res := min + rand.Float64()*(max-min)
	if floor == true {
		res = float64(int64(res))
	}
	return res
}

// RandomWith gives float64 number between range
func RandomWith(min float64, max float64, floor bool, i float64) float64 {
	rand.Seed(time.Now().UTC().UnixNano() * int64(i))
	res := min + rand.Float64()*(max-min)
	if floor == true {
		res = float64(int64(res))
	}
	return res
}

// Map function to map float value in a range
func Map(value float64, start1 float64, stop1 float64, start2 float64, stop2 float64) float64 {
	return (value-start1)/(stop1-start1)*(stop2-start2) + start2
}

// Sigmoid function
func Sigmoid(value float64) float64 {
	return 1 / (1 + math.Exp(-value))
}

//Dsigmoid function
func Dsigmoid(value float64) float64 {
	return value * (1 - value)
}

//Relu function
func Relu(value float64) float64 {
	return math.Max(0, value)
}

//Drelu function
func Drelu(value float64) float64 {
	if value <= 0 {
		return 0
	} else {
		return 1
	}
}
