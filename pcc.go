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
	"sync"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

type fm struct {
	mat64.Matrix
}

type coor struct {
	i [100]int
	j [100]int
}

func (f *coor) SetI(i [100]int) {
	f.i = i
}

func (f coor) I() [100]int {
	return f.i
}

func (f *coor) SetJ(j [100]int) {
	f.j = j
}

func (f coor) J() [100]int {
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
	runtime.GOMAXPROCS(goro)
	c := make([]coor, 1)

	element := coor{}
	var iArr [100]int
	var jArr [100]int
	k := 0

	for i := 0; i < nSets; i++ {
		for j := i; j < nSets; j++ {
			if k <= 99 {
				iArr[k] = i
				jArr[k] = j
			} else {
				element.SetI(iArr)
				element.SetJ(jArr)
				c = append(c, element)
				//fmt.Println(element, "set")
				element = coor{}
				//iArr = nil
				//jArr = nil
				k = 0
				iArr[k] = i
				jArr[k] = j
			}
			k++
			//fmt.Println(i, j, k, " gen")
		}
	}
	//if k >1 {
	element.SetI(iArr)
	element.SetJ(jArr)
	c = append(c, element)
	//}
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

	var wg sync.WaitGroup
	in := make(chan coor, goro*4)
	//quit := make(chan bool, goro)

	singlePCC := func() {
		for {
			select {
			case element := <-in:

				iArr := element.I()
				jArr := element.J()

				for m := 0; m < len(iArr); m++ {
					i := iArr[m]
					j := jArr[m]
					var cv float64
					for k, val := range data.Row(nil, i) {
						cv += data.At(j, k) * val
					}

					cv = cv / (vs[i] * vs[j])
					if (i == 0 && j == 0 && covmat.At(0, 0) == 0.0) || (i+j) > 0 {
						covmat.Set(i, j, cv)
						covmat.Set(j, i, cv)
					}
				}
				wg.Done()
				//case <-quit:
				//	return
			}
		}
	}

	wg.Add(len(c))
	for i := 0; i < goro; i++ {
		go singlePCC()
	}
	//fmt.Println("length: ", len(c))
	for i := 0; i < len(c); i++ {
		in <- c[i]
	}
	wg.Wait()
	//for i := 0; i < goro; i++ {
	//	quit <- true
	//}

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
			fmt.Printf("\t%1.6f", data2.At(i, j))
		}
		fmt.Printf("\n")
	}
}
