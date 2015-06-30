package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

type fm struct {
	mat64.Matrix
}

type coor struct {
	i int
	j int
}

func (f *coor) SetI(i int) {
	f.i = i
}

func (f coor) I() int {
	return f.i
}

func (f *coor) SetJ(j int) {
	f.j = j
}

func (f coor) J() int {
	return f.j
}

func Shift(pToSlice *[]string) string {
	sValue := (*pToSlice)[0]
	*pToSlice = (*pToSlice)[1:len(*pToSlice)]
	return sValue
}

func lcCount(filename string) (lc int, cc int, err error) {
	lc = 0
	cc = 0
	touch := true

	file, err := os.Open(filename)
	if err != nil {
		return
	}
	defer file.Close()

	//load
	br := bufio.NewReaderSize(file, 32768000)
	for {
		line, isPrefix, err1 := br.ReadLine()
		if err1 != nil {
			break
		}
		if isPrefix {
			return
		}

		if touch {
			cc = strings.Count(string(line), "\t")
			cc += 1
			touch = false
		}
		lc++
	}
	return lc, cc, nil

}

func readFile(inFile string, rowName bool) (dataR *mat64.Dense, rName []string, err error) {
	//init
	lc, cc, _ := lcCount(inFile)
	if rowName {
		cc -= 1
	}
	data := mat64.NewDense(lc, cc, nil)
	rName = make([]string, 0)

	//file
	file, err := os.Open(inFile)
	if err != nil {
		return
	}
	defer file.Close()

	//load
	br := bufio.NewReaderSize(file, 32768000)
	r := 0
	for {
		line, isPrefix, err1 := br.ReadLine()
		if err1 != nil {
			break
		}
		if isPrefix {
			return
		}

		str := string(line)
		elements := strings.Split(str, "\t")
		if rowName {
			value := Shift(&elements)
			rName = append(rName, value)
		}
		for c, i := range elements {
			j, _ := strconv.ParseFloat(i, 64)
			data.Set(r, c, j)
		}
		r++
	}
	return data, rName, nil
}

//Multiple threads PCC
func paraCov(data *mat64.Dense, goro int) (covmat *mat64.Dense, err error) {
	nSets, nData := data.Dims()
	if nSets == 0 {
		return mat64.NewDense(0, 0, nil), nil
	}
	runtime.GOMAXPROCS(64)
	c := make([]coor, nSets*nSets)

	for i := 0; i < nSets; i++ {
		for j := i; j < nSets; j++ {
			element := coor{}
			element.SetI(i)
			element.SetJ(j)
			c = append(c, element)
		}
	}

	covmat = mat64.NewDense(nSets, nSets, nil)
	means := make([]float64, nSets)
	//var sqrt
	vs := make([]float64, nSets)
	for i := range means {
		means[i] = floats.Sum(data.Row(nil, i)) / float64(nData)
		var element float64
		for j, _ := range data.Row(nil, i) {
			data.Set(i, j, data.At(i, j)-means[i])
			element += data.At(i, j) * data.At(i, j)
		}
		vs[i] = math.Sqrt(element)
	}

	in := make(chan coor)
	quit := make(chan bool)

	singlePCC := func() {
		for {
			select {
			case element := <-in:

				i := element.I()
				j := element.J()
				var cv float64

				for k, val := range data.Row(nil, i) {
					cv += data.At(j, k) * val
				}

				cv = cv / (vs[i] * vs[j])
				covmat.Set(i, j, cv)
				covmat.Set(j, i, cv)

			case <-quit:
				return
			}
		}
	}

	for i := 0; i < goro; i++ {
		go singlePCC()
	}

	for i := 0; i < len(c); i++ {
		in <- c[i]
	}

	for i := 0; i < goro; i++ {
		quit <- true
	}

	return covmat, nil
}

func main() {
	var inFile *string = flag.String("i", "test.txt", "tab delimited matrix")
	var inThreads *int = flag.Int("p", 1, "number of threads")
	flag.Parse()
	data, rName, _ := readFile(*inFile, true)
	data2, _ := paraCov(data, *inThreads)
	_, nCol := data2.Caps()
	for i := range rName {
		fmt.Printf("%v", rName[i])
		for j := 0; j < nCol; j++ {
			fmt.Printf("\t%1.2f", data2.At(i, j))
		}
		fmt.Printf("\n")
	}
}
