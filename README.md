# SoftImpute-ALS with python

This is a python implementation of the softImpute, ALS, and softImpute-ALS algorithms from the paper *Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares* by Hastie, Trevor, Rahul Mazumder, Jason Lee, and Reza Zadeh. 

Please note that it is an naive implementation of the sudo code in the paper, and therefore does not make use of the sparse and low rank representation of the matrix X. We also diretly used packages like numpy.
This would provide an unfair advantage to methods like softImpute, whose structure is quite simple and the bottleneck step is the SVD. Therefore, you may find the experiement result does not perfectly agree with the paper.

This implementation referenced several existing repositories, and also used generative AI to assisst coding.

Reference:
- An existing implementation of simple ALS method: https://github.com/tansey/netflix_als
- An existing implementation of softImpute-ALS: https://github.com/travisbrady/py-soft-impute
- R package: https://github.com/cran/softImpute
- paper: http://arxiv.org/abs/1410.2596
- Another python implementation on these functions: https://github.com/modularity/softImpute-ALS
