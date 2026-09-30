"""Optional CUDA scorer for the frozen numeric sklearn histogram tree ensemble.

Retrieval and feature extraction remain on CPU. This is not a GPU trainer.
Require parity on exported reference probabilities before using for inference.
"""
import numpy as np

CUDA=r'''
extern "C" __global__ void score(
 const float* x, const int n, const int width, const int nt,
 const int* roots, const int* feature, const double* threshold,
 const int* left, const int* right, const unsigned char* leaf,
 const unsigned char* missing_left, const double* value,
 const double baseline, double* out) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;
 if(i>=n)return;
 double sum=baseline;
 for(int t=0;t<nt;t++){
  int node=roots[t];
  while(!leaf[node]){
   double v=(double)x[i*width+feature[node]];
   bool go_left=isnan(v) ? missing_left[node] : v<=threshold[node];
   node=go_left ? left[node] : right[node];
  }
  sum+=value[node];
 }
 out[i]=1.0/(1.0+exp(-sum));
}
'''

def export(model,path):
    if len(model.classes_)!=2 or list(model.classes_)!=[0,1]:raise ValueError('Only binary 0/1 classifier supported')
    arrays={k:[] for k in ['feature','threshold','left','right','leaf','missing_left','value']};roots=[];offset=0
    for iteration in model._predictors:
        if len(iteration)!=1:raise ValueError('Only binary ensemble supported')
        nodes=iteration[0].nodes
        if np.any(nodes['is_categorical']):raise ValueError('Categorical nodes unsupported')
        roots.append(offset)
        for target,source in [('feature','feature_idx'),('threshold','num_threshold'),('left','left'),('right','right'),('leaf','is_leaf'),('missing_left','missing_go_to_left'),('value','value')]:
            a=nodes[source].copy()
            if target in ('left','right'):a=a.astype(np.int32)+offset
            arrays[target].append(a)
        offset+=len(nodes)
    packed={k:np.concatenate(v).astype(np.int32 if k in ('feature','left','right') else np.uint8 if k in ('leaf','missing_left') else np.float64) for k,v in arrays.items()}
    np.savez_compressed(path,**packed,roots=np.asarray(roots,dtype=np.int32),baseline=np.asarray(model._baseline_prediction).reshape(-1),width=model.n_features_in_)

class GPUMatcher:
    def __init__(self,path):
        import cupy as cp
        self.cp=cp
        with np.load(path) as f:
            self.width=int(f['width']);self.baseline=float(f['baseline'][0]);self.nt=len(f['roots'])
            self.arrays=[cp.asarray(f[k]) for k in ['roots','feature','threshold','left','right','leaf','missing_left','value']]
        self.kernel=cp.RawKernel(CUDA,'score')
    def predict_proba(self,x):
        cp=self.cp;x=cp.asarray(x,dtype=cp.float32,order='C')
        if x.ndim!=2 or x.shape[1]!=self.width:raise ValueError('Feature shape mismatch')
        out=cp.empty(len(x),dtype=cp.float64)
        if len(x):self.kernel(((len(x)+255)//256,),(256,),(x,np.int32(len(x)),np.int32(self.width),np.int32(self.nt),*self.arrays,np.float64(self.baseline),out))
        positive=cp.asnumpy(out)
        return np.column_stack((1-positive,positive))
