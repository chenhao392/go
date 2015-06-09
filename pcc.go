import "github.com/gonum/matrix/mat64"
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
	covmat := mat64.NewDense(nSets, nSets, nil)
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
}
