package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/gonum/matrix/mat64"
)

type fm struct {
	mat64.Matrix
}

type geneDis struct {
	index    int
	value    []float64
	isFisher bool
}

type benchCase struct {
	path  string
	bench map[int]string
	k     int
	isRRA bool
}

type Slice struct {
	sort.Interface
	idx []int
}

func (f *geneDis) SetIndex(i int) {
	f.index = i
}

func (f geneDis) GetIndex() int {
	return f.index
}

func (f *geneDis) SetValue(value []float64) {
	f.value = value
}

func (f geneDis) GetValue() []float64 {
	return f.value
}

func (f *geneDis) SetMethod(isFisher bool) {
	f.isFisher = true
}

func (f geneDis) GetMethod() bool {
	return f.isFisher
}

func (f *benchCase) SetPath(path string) {
	f.path = path
}

func (f benchCase) GetPath() string {
	return f.path
}

func (f *benchCase) SetBench(bench map[int]string) {
	f.bench = bench
}

func (f benchCase) GetBench() map[int]string {
	return f.bench
}

func (f *benchCase) SetK(k int) {
	f.k = k
}

func (f benchCase) GetRRA() bool {
	return f.isRRA
}
func (f *benchCase) SetRRA(isRRA bool) {
	f.isRRA = isRRA
}

func (f benchCase) GetK() int {
	return f.k
}

func (s Slice) Swap(i, j int) {
	s.Interface.Swap(i, j)
	s.idx[i], s.idx[j] = s.idx[j], s.idx[i]
}

func NewSlice(n sort.Interface) *Slice {
	s := &Slice{Interface: n, idx: make([]int, n.Len())}
	for i := range s.idx {
		s.idx[i] = i
	}
	return s
}

func NewFloat64Slice(n ...float64) *Slice {
	return NewSlice(sort.Float64Slice(n))
}

func Shift(pToSlice *[]string) string {
	sValue := (*pToSlice)[0]
	*pToSlice = (*pToSlice)[1:len(*pToSlice)]
	return sValue
}

// this function convert the ordered index to actural ranks
func order2float64(intArray []int) []float64 {
	floatArray := make([]float64, len(intArray))
	for i := 0; i < len(intArray); i++ {
		floatArray[intArray[i]] = float64(i + 1)
	}
	return floatArray
}

func Choose(N, K int64) int64 {
	var combi int64 = 1
	for i := N; i > (N - K); i-- {
		combi *= i
	}
	for i := K; i >= 1; i-- {
		combi /= i
	}

	return combi
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

func readMatrix(inFile string, rowName bool) (dataR *mat64.Dense, gene2index2 map[string]int, index2gene2 map[int]string, err error) {
	//init
	lc, cc, _ := lcCount(inFile)
	if rowName {
		cc -= 1
	}
	data := mat64.NewDense(lc, cc, nil)
	gene2index := make(map[string]int, lc)
	index2gene := make(map[int]string, lc)
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
			gene2index[value] = r
			index2gene[r] = value
		}
		for c, i := range elements {
			j, _ := strconv.ParseFloat(i, 64)
			data.Set(r, c, j)
		}
		r++
	}
	return data, gene2index, index2gene, nil
}

func readBenchmark(inFile string, tNum int, sNum int64, gene2index map[string]int) (gene2 []int, benchSet2 map[string]map[int]string, testSet2 map[string]map[int]string, err error) {
	rand.Seed(sNum)
	pathway := make(map[string][]string)
	gene := make([]int, 0)
	benchSet := make(map[string]map[int]string)
	testSet := make(map[string]map[int]string)
	//file
	file, err := os.Open(inFile)
	if err != nil {
		return
	}
	defer file.Close()

	//load
	br := bufio.NewReaderSize(file, 32768)
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
		pathway[elements[0]] = append(pathway[elements[0]], elements[1])
		gene = append(gene, gene2index[elements[1]])
	}

	//benchmark
	for path, genes := range pathway {
		benchLength := 2 * len(genes) / 3
		testLength := len(genes) - benchLength
		for i := 0; i < tNum; i++ {
			b := make(map[int]string, benchLength)
			t := make(map[int]string, testLength)
			perm := rand.Perm(len(genes))
			name := path + "_" + strconv.Itoa(i)
			for j, _ := range perm {
				index, err := gene2index[genes[perm[j]]]

				if !err {
					panic(err)
				}
				if j < benchLength {
					b[index] = ""
				} else {
					t[index] = ""
				}
			}

			benchSet[name] = b
			testSet[name] = t
		}
	}

	return gene, benchSet, testSet, nil
}

// either fisher or mg transformation
func mergeMatrix(matrix1 *mat64.Dense, matrix2 *mat64.Dense, gene2index1 map[string]int, gene2index2 map[string]int, goro int, isFisher bool) (mergeMatrixOut *mat64.Dense, gene2indexOut map[string]int, index2geneOut map[int]string, err error) {
	tM1, _ := transformMatrix(matrix1, goro, isFisher)
	tM2, _ := transformMatrix(matrix2, goro, isFisher)
	gene2index := make(map[string]int, 0)
	index2gene := make(map[int]string, 0)

	i := 0
	for gene, _ := range gene2index1 {
		gene2index[gene] = i
		index2gene[i] = gene
		i++
	}
	for gene, _ := range gene2index2 {
		_, exists := gene2index[gene]
		if !exists {
			gene2index[gene] = i
			index2gene[i] = gene
			i++
		}
	}

	mergeMatrix := mat64.NewDense(len(gene2index), len(gene2index), nil)
	for i := 0; i < len(index2gene); i++ {
		for j := i; j < len(index2gene); j++ {
			t1, e1 := gene2index1[index2gene[i]]
			t2, e2 := gene2index1[index2gene[j]]
			t3, e3 := gene2index2[index2gene[i]]
			t4, e4 := gene2index2[index2gene[j]]

			var v1 float64
			var v2 float64

			if e1 && e2 {
				v1 = tM1.At(t1, t2)
			} else {
				if isFisher {
					v1 = -1.39
				} else {
					v1 = 0
				}
			}

			if e3 && e4 {
				v2 = tM2.At(t3, t4)
			} else {
				if isFisher {
					v2 = -1.39
				} else {
					v2 = 0
				}
			}

			mergeMatrix.Set(i, j, (v1 + v2))
			mergeMatrix.Set(j, i, (v1 + v2))
		}
	}
	return mergeMatrix, gene2index, index2gene, nil
}

func transformMatrix(inMatrix *mat64.Dense, goro int, isFisher bool) (outMatrix2 *mat64.Dense, err error) {
	nRow, nCol := inMatrix.Caps()
	outMatrix := mat64.NewDense(nRow, nCol, nil)
	c := make([]geneDis, nRow)
	element := geneDis{}

	for i := 0; i < nRow; i++ {

		element.SetIndex(i)
		element.SetValue(inMatrix.RawRowView(i))
		element.SetMethod(true)
		c[i] = element
	}

	var wg sync.WaitGroup
	in := make(chan geneDis, goro*40)
	runtime.GOMAXPROCS(goro)

	singleTrans := func() {
		for {
			select {
			case element := <-in:
				index := element.GetIndex()
				array := element.GetValue()
				isFisher := element.GetMethod()
				tArray := make([]float64, len(array))

				//sign flipped, as the single matrix method use pcc.
				// It helps when rank all these results.
				if isFisher {
					for i := 0; i < len(array); i++ {
						tArray[i] = 2 * math.Log(array[i]+0.000001)
					}
				} else {
					for i := 0; i < len(array); i++ {
						tArray[i] = math.Log(array[i] / (1 - array[i]))
					}
				}
				outMatrix.SetRow(index, tArray)

				wg.Done()
			}
		}
	}

	wg.Add(len(c))
	for i := 0; i < goro; i++ {
		go singleTrans()
	}
	//fmt.Println("length: ", len(c))
	for i := 0; i < len(c); i++ {
		in <- c[i]
	}
	wg.Wait()

	return outMatrix, nil
}

//Multiple threads sort
func paraSort(data *mat64.Dense, gene []int, goro int) (covmat *mat64.Dense, err error) {
	nSets, _ := data.Dims()
	covmat = mat64.NewDense(nSets, nSets, nil)

	if nSets == 0 {
		return mat64.NewDense(0, 0, nil), nil
	}
	c := make([]geneDis, 1)
	element := geneDis{}

	for i := 0; i < len(gene); i++ {

		element.SetIndex(gene[i])
		element.SetValue(data.RawRowView(gene[i]))
		c = append(c, element)
	}

	var wg sync.WaitGroup
	in := make(chan geneDis, goro*40)
	runtime.GOMAXPROCS(goro)

	//quit := make(chan bool, goro)

	singleSort := func() {
		for {
			select {
			case element := <-in:
				index := element.GetIndex()
				array := element.GetValue()
				order := NewFloat64Slice(array...)

				sort.Sort(sort.Reverse(order))
				covmat.SetRow(index, order2float64(order.idx))

				wg.Done()

			}
		}
	}

	wg.Add(len(c))
	for i := 0; i < goro; i++ {
		go singleSort()
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

func prCurveRRA(rankMatrix1 *mat64.Dense, rankMatrix2 *mat64.Dense, goro int, capRank1 int, capRank2 int, setLength1 int, setLength2 int, bench map[int]string, test map[int]string, index2gene map[int]string, index2gene2 map[int]string, gene2index2 map[string]int) (pr2 [][]string, err error) {

	pr := make([][]string, 0)

	//bench
	//potential problem here.... set length assummed to be similar, and set 1 is larger
	rankEst := make([]float64, setLength1)
	// test about ranks
	if capRank1 > capRank2 {
		capRank2 = capRank1
	} else {
		capRank1 = capRank2
	}

	c := make([]int, 0)
	for i := 0; i < setLength1; i++ {
		c = append(c, i)
	}

	var wg sync.WaitGroup
	in := make(chan int, goro*40)
	runtime.GOMAXPROCS(goro)

	//quit := make(chan bool, goro)

	singleMin := func() {
		for {
			select {
			case i := <-in:

				//normalized ranks
				genes := make([]float64, len(bench)*2)
				for j := 0; j < len(genes); j++ {
					genes[j] = 1
				}
				k := 0
				for j, _ := range bench {
					if rankMatrix1.At(j, i) > float64(capRank1) {
					} else {
						genes[k] = rankMatrix1.At(j, i) / float64(setLength1)
					}
					k++
				}
				for j, _ := range bench {
					j2 := gene2index2[index2gene[j]]
					i2 := gene2index2[index2gene[i]]
					if rankMatrix2.At(j2, i2) > float64(capRank2) {
					} else {
						genes[k] = rankMatrix2.At(j2, i2) / float64(setLength2)
					}
					k++
				}
				//sort
				geneSlice := NewFloat64Slice(genes...)
				sort.Sort(geneSlice)
				order := geneSlice.idx
				for _, n := range order {
					combi := float64(Choose(int64(len(genes)), int64(n+1)))
					v1 := math.Pow(genes[n], float64(n+1))
					var v2 float64
					v2 = 1
					if genes[n] != 1 {
						v2 = math.Pow((1 - genes[n]), float64(len(genes)-n-1))
					}
					value := combi * v1 * v2
					if rankEst[i] == 0 {
						rankEst[i] = value
						//fmt.Println(n, combi, genes[n], value, v1, v2)
					} else if rankEst[i] > value {
						rankEst[i] = value
					}
				}

				wg.Done()

			}
		}
	}

	wg.Add(len(c))
	for i := 0; i < goro; i++ {
		go singleMin()
	}
	//fmt.Println("length: ", len(c))
	for i := 0; i < len(c); i++ {
		in <- c[i]
	}
	wg.Wait()

	//test
	p := 0
	tp := 0
	rankEstSlice := NewFloat64Slice(rankEst...)
	sort.Sort(rankEstSlice)
	for _, i := range rankEstSlice.idx {
		_, inTest := test[i]
		_, inBench := bench[i]
		if !inBench {
			p++
		}
		if inTest {
			tp++
			//fmt.Println(path, p, tp)
			arr := make([]string, 3)
			arr[0] = strconv.FormatFloat(float64(tp)/float64(p), 'f', 3, 64)
			arr[1] = strconv.FormatFloat(float64(tp)/float64(len(test)), 'f', 3, 64)
			arr[2] = index2gene[i]
			pr = append(pr, arr)
		}
	}

	return pr, nil

}

func prCurve(rankMatrix *mat64.Dense, isRRA bool, capRank int, setLength int, bench map[int]string, test map[int]string, index2gene map[int]string) (pr2 [][]string, err error) {
	pr := make([][]string, 0)

	//bench
	rankEst := make([]float64, setLength)
	if isRRA {
		for i := 0; i < setLength; i++ {
			//normalized ranks
			genes := make([]float64, len(bench))
			for j := 0; j < len(genes); j++ {
				genes[j] = 1
			}
			k := 0
			for j, _ := range bench {
				if rankMatrix.At(j, i) > float64(capRank) {
					//genes[k] = rankMatrix.At(j, i) / float64(setLength)
				} else {
					genes[k] = rankMatrix.At(j, i) / float64(setLength)
				}
				k++
			}
			//sort
			geneSlice := NewFloat64Slice(genes...)
			sort.Sort(geneSlice)
			order := geneSlice.idx
			for _, n := range order {
				combi := float64(Choose(int64(len(genes)), int64(n+1)))
				v1 := math.Pow(genes[n], float64(n+1))
				var v2 float64
				v2 = 1
				if genes[n] != 1 {
					v2 = math.Pow((1 - genes[n]), float64(len(genes)-n-1))
				}
				value := combi * v1 * v2
				if rankEst[i] == 0 {
					rankEst[i] = value
					//fmt.Println(n, combi, genes[n], value, v1, v2)
				} else if rankEst[i] > value {
					rankEst[i] = value
				}
			}

		}
	} else {
		for i := 0; i < setLength; i++ {
			for j, _ := range bench {
				ele := int(rankMatrix.At(j, i))
				if ele <= capRank {
					rankEst[i] += rankMatrix.At(j, i)
				} else {
					rankEst[i] += float64(capRank)
				}
			}
		}
	}
	//test
	p := 0
	tp := 0
	rankEstSlice := NewFloat64Slice(rankEst...)
	sort.Sort(rankEstSlice)
	for _, i := range rankEstSlice.idx {
		_, inTest := test[i]
		_, inBench := bench[i]
		if !inBench {
			p++
		}
		if inTest {
			tp++
			//fmt.Println(path, p, tp)
			arr := make([]string, 3)
			arr[0] = strconv.FormatFloat(float64(tp)/float64(p), 'f', 3, 64)
			arr[1] = strconv.FormatFloat(float64(tp)/float64(len(test)), 'f', 3, 64)
			arr[2] = index2gene[i]
			pr = append(pr, arr)
		}
	}
	return pr, nil
}

func prCurveEst(rankMatrix *mat64.Dense, isRRA bool, goro int, capRank int, setLength int, benchSet map[string]map[int]string, index2gene map[int]string) (kCube2 map[string][]float64, err error) {
	kCube := make(map[string][]float64, len(benchSet))
	for path, _ := range benchSet {
		kCube[path] = make([]float64, capRank)
	}
	c := make([]benchCase, len(benchSet)*(capRank))
	element := benchCase{}
	count := 0
	for path, bench := range benchSet {
		for k := 1; k <= capRank; k++ {
			element.SetPath(path)
			element.SetBench(bench)
			element.SetK(k)
			element.SetRRA(isRRA)
			c[count] = element
			count++
		}
	}

	var wg sync.WaitGroup
	in := make(chan benchCase, goro*40)
	runtime.GOMAXPROCS(goro)

	singleBench := func() {
		for {
			select {
			case element := <-in:
				path := element.GetPath()
				inBench := element.GetBench()
				isRRA := element.GetRRA()
				keys := make([]int, 0, len(inBench))
				for key := range inBench {
					keys = append(keys, key)
				}
				sort.Ints(keys)

				bench := make(map[int]string)
				test := make(map[int]string)
				halfLength := len(inBench) / 2
				for i, key := range keys {
					if i < halfLength {
						bench[key] = ""
					} else {
						test[key] = ""
					}
				}
				//bench := inBench[0:halfLength]
				//test := inBench[(halfLength + 1):]
				k := element.GetK()
				rankEst := make([]float64, setLength)
				if isRRA {
					for i := 0; i < setLength; i++ {
						//normalized ranks
						genes := make([]float64, len(bench))
						for j := 0; j < len(genes); j++ {
							genes[j] = 1
						}
						m := 0
						for j, _ := range bench {
							if rankMatrix.At(j, i) > float64(m) {
								//genes[k] = rankMatrix.At(j, i) / float64(setLength)
							} else {
								genes[m] = rankMatrix.At(j, i) / float64(setLength)
							}
							m++
						}
						//sort
						geneSlice := NewFloat64Slice(genes...)
						sort.Sort(geneSlice)
						order := geneSlice.idx
						for _, n := range order {
							combi := float64(Choose(int64(len(genes)), int64(n+1)))
							v1 := math.Pow(genes[n], float64(n+1))
							var v2 float64
							v2 = 1
							if genes[n] != 1 {
								v2 = math.Pow((1 - genes[n]), float64(len(genes)-n-1))
							}
							value := combi * v1 * v2
							if rankEst[i] == 0 {
								rankEst[i] = value
								//fmt.Println(n, combi, genes[n], value, v1, v2)
							} else if rankEst[i] > value {
								rankEst[i] = value
							}
						}
					}
				} else {
					for i := 0; i < setLength; i++ {
						for j, _ := range bench {
							ele := int(rankMatrix.At(j, i))
							if ele <= k {
								rankEst[i] += rankMatrix.At(j, i)
							} else {
								rankEst[i] += float64(k)
							}
						}
					}
				}
				//test
				p := 0
				tp := 0
				pr := make([][]float64, len(test))
				rankEstSlice := NewFloat64Slice(rankEst...)
				sort.Sort(rankEstSlice)
				for _, i := range rankEstSlice.idx {
					_, inTest := test[i]
					_, inBench := bench[i]
					if !inBench {
						p++
					}
					if inTest {
						tp++
						//fmt.Println(path, p, tp)
						pr[tp-1] = make([]float64, 2)
						pr[tp-1][0] = float64(tp) / float64(p)
						pr[tp-1][1] = float64(tp) / float64(len(test))
					}
				}

				var aupr float64
				for j := 1; j < len(pr); j++ {
					aupr += (pr[j-1][0] + pr[j][0]) * (pr[j][1] - pr[j-1][1])
					//fmt.Println(path, k, pr[j][0], pr[j][1], aupr)
				}
				if aupr > 1 {
					aupr = 1
				}
				//fmt.Println(path, k, aupr)
				kCube[path][k-1] = aupr

				wg.Done()
			}
		}
	}

	wg.Add(len(c))
	for i := 0; i < goro; i++ {
		go singleBench()
	}
	//fmt.Println("length: ", len(c))
	for i := 0; i < len(c); i++ {
		in <- c[i]
	}
	wg.Wait()

	return kCube, nil
}

func kSelect(kCube map[string][]float64) (kSet2 map[string]int, err error) {
	kSet := make(map[string]int)
	for path, aupr := range kCube {
		auprSlice := NewFloat64Slice(aupr...)
		sort.Sort(sort.Reverse(auprSlice))
		//fmt.Println(path, auprSlice.idx[0])
		kSet[path] = auprSlice.idx[0] + 1
	}
	return kSet, nil
}

//https://en.wikipedia.org/wiki/Trapezoidal_rule
// integrate with non-uniform rule
func auprCal(pr map[string][][]string) (aupr2 map[string]float64, err error) {
	aupr := make(map[string]float64, len(pr))
	for path, arr := range pr {

		for j := 1; j < len(arr); j++ {
			temp := make([]float64, 4)
			temp[0], _ = strconv.ParseFloat(arr[j][0], 64)
			temp[1], _ = strconv.ParseFloat(arr[j-1][0], 64)
			temp[2], _ = strconv.ParseFloat(arr[j][1], 64)
			temp[3], _ = strconv.ParseFloat(arr[j-1][1], 64)

			aupr[path] += (temp[0] + temp[1]) * (temp[2] - temp[3])
		}
		aupr[path] /= 2
	}

	return aupr, nil
}

func main() {
	var inMatrix *string = flag.String("m", "test_rank_matrix.txt", "tab delimited matrix")
	var inMatrix2 *string = flag.String("m2", "test_rank_matrix.txt", "tab delimited matrix")
	var inBenchmark *string = flag.String("b", "test_rank_bench.txt", "tab delimited benchmark")
	var inThreads *int = flag.Int("p", 1, "number of threads")
	var randNum *int64 = flag.Int64("r", 392, "rand seed")
	var testNum *int = flag.Int("t", 10, "number of tests per pathway")
	var capRank *int = flag.Int("c", 200, "cap rank for majority votes")
	var isAUPR *bool = flag.Bool("a", true, "aupr output")
	var isSingle *bool = flag.Bool("s", true, "single or combine matrix (fisher|mg)")
	var combiMethod *string = flag.String("f", "fisher", "RRA, fisher or mg")

	flag.Parse()

	pr := make(map[string][][]string)
	var kSet map[string]int
	var kSet1 map[string]int
	var kSet2 map[string]int
	if *isSingle {
		matrix, gene2index, index2gene, _ := readMatrix(*inMatrix, true)
		genes, benchSet, testSet, _ := readBenchmark(*inBenchmark, *testNum, *randNum, gene2index)
		rankMatrix, _ := paraSort(matrix, genes, *inThreads)
		_, nCol := rankMatrix.Caps()
		isRRA := strings.EqualFold(*combiMethod, "RRA")
		kCube, _ := prCurveEst(rankMatrix, isRRA, *inThreads, *capRank, nCol, benchSet, index2gene)
		kSet, _ = kSelect(kCube)
		for path, k := range kSet {
			prEle, _ := prCurve(rankMatrix, isRRA, k, nCol, benchSet[path], testSet[path], index2gene)
			pr[path] = prEle
			//fmt.Println(path, kCube[path])
		}
	} else {
		matrix1, gene2index1, index2gene1, _ := readMatrix(*inMatrix, true)
		matrix2, gene2index2, index2gene2, _ := readMatrix(*inMatrix2, true)

		if strings.EqualFold(*combiMethod, "RRA") {
			genes1, benchSet1, testSet, _ := readBenchmark(*inBenchmark, *testNum, *randNum, gene2index1)
			genes2, benchSet2, _, _ := readBenchmark(*inBenchmark, *testNum, *randNum, gene2index2)
			rankMatrix1, _ := paraSort(matrix1, genes1, *inThreads)
			rankMatrix2, _ := paraSort(matrix2, genes2, *inThreads)
			_, nCol1 := rankMatrix1.Caps()
			_, nCol2 := rankMatrix2.Caps()
			kCube1, _ := prCurveEst(rankMatrix1, true, *inThreads, *capRank, nCol1, benchSet1, index2gene1)
			kCube2, _ := prCurveEst(rankMatrix2, true, *inThreads, *capRank, nCol2, benchSet2, index2gene2)

			kSet1, _ = kSelect(kCube1)
			kSet2, _ = kSelect(kCube2)
			for path, k1 := range kSet1 {
				prEle, _ := prCurveRRA(rankMatrix1, rankMatrix2, *inThreads, k1, kSet2[path], nCol1, nCol2, benchSet1[path], testSet[path], index2gene1, index2gene2, gene2index2)
				pr[path] = prEle
			}

		} else {
			isFisher := false
			if strings.EqualFold(*combiMethod, "fisher") {
				isFisher = true
			}

			mergeMatrix, gene2index, index2gene, _ := mergeMatrix(matrix1, matrix2, gene2index1, gene2index2, *inThreads, isFisher)
			genes, benchSet, testSet, _ := readBenchmark(*inBenchmark, *testNum, *randNum, gene2index)
			rankMatrix, _ := paraSort(mergeMatrix, genes, *inThreads)
			_, nCol := rankMatrix.Caps()
			//pr, _ = prCurve(rankMatrix, *capRank, nCol, benchSet, testSet, index2gene)
			kCube, _ := prCurveEst(rankMatrix, true, *inThreads, *capRank, nCol, benchSet, index2gene)
			kSet, _ = kSelect(kCube)
			for path, k := range kSet {
				prEle, _ := prCurve(rankMatrix, true, k, nCol, benchSet[path], testSet[path], index2gene)
				pr[path] = prEle
			}
		}
	}
	//genes, _, _, _ := readBenchmark(*inBenchmark, 10, 392, gene2index)

	if *isAUPR {
		aupr, _ := auprCal(pr)
		for path, v := range aupr {
			fmt.Printf("%s", path)
			if !*isSingle {
				fmt.Printf("\t%1.2f\t%d\t%d\n", v, kSet1[path], kSet2[path])
			} else {
				fmt.Printf("\t%1.2f\t%d\n", v, kSet[path])
			}
		}
	} else {
		for path, v := range pr {
			for j := 0; j < len(v); j++ {
				fmt.Printf("%s", path)
				for k := 0; k < len(v[j]); k++ {
					fmt.Printf("\t%s", v[j][k])
				}
				if !*isSingle {
					fmt.Printf("\t%1.2f\t%d\t%d\n", v, kSet1[path], kSet2[path])
				} else {
					fmt.Printf("\t%d\n", kSet[path])
				}
			}
		}
	}

}
