# pycsminwel
This is a locol minimization algorithm. Uses a quasi-Newton method with BFGS
update of the estimated inverse hessian. It is robust against certain
pathologies common on likelihood functions. It attempts to be robust against
"cliffs", i.e. hyperplane discontinuities, though it is not really clear
whether what it does in such cases succeeds reliably.

This algorithm is somewhat more robust, apparently, than the stock optimization 
programs that do about the same thing. The minimizer can negotiate discontinuous 
"cliffs" without getting stuck.

## Author
All the functions in csminwel.py were created by Christopher Sims in 1996.
They are my translations to python from the author's original MATLAB files.

The originals can be found in:
http://sims.princeton.edu/yftp/optimize/

I kept the author's original variable names, so it is easier to compare 
this code with his. Also, this code is not very pythonic, it is very 
matlaby. I need a better understanding of the algorithm before making 
it more pythonic.