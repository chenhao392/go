package main
import ("github.com/gonum/matrix/mat64"
	"bufio"
	"fmt"
	"flag"
	"strings"
	)


func main(){
	var inFile *string=flag("i","inMatrix.txt","tab delimited matrix")
	data := raedFile(inFile)
	//fmt.print(data)
}
func readFile(inFile string)(data [][]float64,error){
	file, err := os.Open(inFile)
	if err != nil {
		return
	}
	defer file.Close()
	br :=bufio.NewReader(file)
	data = [][]float64
	for{
		line,isPrefix,err1 := br.ReadLine()
		if err1 != nil{
			break
		}
		if isPrefix{
			return
		}
		str=string(line)
		elements :=strings.Split(str,"\t")
		data=append(data,elements)
	}
	return data
}


// input: 2D array data in float64
//output: pointer to * mat64.Dense to a  struc Dense
//Dense is a dense matrix representation.
//type Dense struct {
// 	mat blas64.General
//  	capRows, capCols int
//}
func cov(data ...[]float64) (*mat64.Dense, error) {
	nSets := len(data)
	if nSets == 0 {
		return mat64.NewDense(0, 0, nil), nil
	}
	nData := len(data[0])
	for i := range data {
		if len(data[i]) != nData {
			return nil, errors.New("cov: datasets have unequal size")
		}
	}
	//new Dense struc with row and columns defined as nSets, nil is for empty mat
	covmat := mat64.NewDense(nSets, nSets, nil)
	//slice function make to generate an array slice with length nSets
	means := make([]float64, nSets)
	for i := range means {
		means[i] = floats.Sum(data[i]) / float64(nData)
	}
	for i := 0; i < nSets; i++ {
		for j := i; j < nSets; j++ {
			var cv float64
			meanI := means[i]
			meanJ := means[j]
			invData := 1 / float64(nData-1)
			for k, val := range data[i] {
				cv += invData * (val - meanI) * (data[j][k] - meanJ)
			}
			covmat.Set(i, j, cv)
			covmat.Set(j, i, cv)
		}
	}
	return covmat
}
