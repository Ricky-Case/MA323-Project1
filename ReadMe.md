# Programming Project 1
Due: 11am, Monday, September 16, 2024.

Write the scripts for the following problems in jupyter notebook and combine them as a single PDF file.

## Part 1
(1pt)
- Write a function vec_add_plot(v1, v2) that visualizes the addition of the two input vectors v1 and v2 in R2.

- If at least one of the inputs is not a two-dimensional vector, then print out an error message.

- Test the function with the vectors [4, 2] and [-1, 2].
(Hint: import numpy and matplotlib packages, and refer to quiver https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html)

## Part 2
(2pt)
- Write a function check_orth(v1, v2, tol), where v1 and v2 are column vectors of the same size and tol is a scalar value strictly larger than θ.

- The output should be 1 (true) if the angle θ between v1 and v2 is within tol of π/2.
i.e. |π/2 − θ| < tol.

- The output should be 0 (false) otherwise.
Note: cos θ = (v1 * v2) / (||v1|| * ||v2||).
You may assume that v1 and v2 are column vectors of the same size, and that tol is a positive scalar.

- Include one test case with v1 = [5, 4, 4, 5, 1, 4, 1]^T and v2 = [−1, −2, 1, −2, 3, 1, −5]^T and tol = 1e-3.

## Part 3
(2pt)
- Read the data set A.csv as a matrix A ∈ R^(30 * 6) via:
> from numpy import genfromtxt
> A = genfromtxt(’A.csv’, delimiter=’,’)

- Compute the SVD of A and find:
1. The third right singular vector
2. The second singular value
3. The fourth left singular vector
4. The rank of A
