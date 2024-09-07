import numpy as np
import matplotlib.pyplot as mpl

# (Hint: import numpy and matplotlib packages, and refer to quiver https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html)

# Visualize the addition of two input vectors "v1" and "v2" in 2D.
def vec_add_plot(v1, v2):
    origin = np.array([[0,0], [0,0]])
    
    if (len(v1) != 2 or len(v2) != 2):
        print("Error: Please enter two vectors of length 2.")
        return
    else:
        print("Computing ", v1, " + ", v2, "...\n")
    
    vectors = np.array([v1, v2])
    result = np.array(v1) + np.array(v2)
    
    print("Result: ", result, "\n\n")
    print("With origin: ", origin)
    
    # Visualize the vector addition with matplotlib.
    # mplib.plot(vectors[0], "r-", label = "Vector 1")
    # mplib.plot(vectors[1], "b-", label = "Vector 2")
    # mplib.plot(result, "g-", label = "Resulting Vector")
    mpl.quiver(*origin, vectors[:,0], vectors[:,1], color = ['g', 'b'], angles = 'xy', scale_units = 'xy', scale = 1)
    
    mpl.quiver(*origin, result[0], result[1], angles = 'xy', scale_units = 'xy', scale = 1)
    
    mpl.xlim(-4, 4)
    mpl.ylim(-4, 4)
    mpl.gca().set_aspect("equal")
    
    mpl.title("Vector Addition")
    mpl.xlabel("X-Axis")
    mpl.ylabel("Y-Axis")
    
    mpl.show()


# Check the orthogonality of two column vectors "v1" and "v2" of the same length, using tol as a scalar value strictly larger than 0.
def check_ortho(v1, v2, tol):
    if (len(v1) != len(v2)):
        print("Error: Vector Dimension Mismatch!\n",
              "Please enter two vectors which have equivalent dimensions.")
        return
    elif (tol <= 0):
        print("Error: Integer provided must be breater than 0.")
        return

    cv1 = np.array(v1).T
    cv2 = np.array(v2).T
    
    print("Column Vector 1:\n", cv1, "\n")
    print("Column Vector 2:\n", cv2, "\n")
    
    if((np.dot(cv1, cv2) < (0 - tol)) or (np.dot(cv1, cv2) > (0 + tol))):
        return False
    else:
        return True


################
#### PART 1 ####
################


print("Part 1\n\n")

vector1 = [4, 2]
vector2 = [-1, 2]
vec_add_plot(vector1, vector2)


################
#### PART 2 ####
################

print("Part 2\n\n")

columnVector1 = [5, 4, 4, 5, 1, 4, 1]
columnVector2 = [-1, -2, 1, -2, 3, 1, -5]
tolVal = 0.001

if(check_ortho(columnVector1, columnVector2, tolVal)):
    print("1 - These vectors are orthogonal.\n")
else:
    print("0 - These vectors are not orthogonal.\n")


columnVector1 = [2, 18]
columnVector2 = [3/2, -1/6]
tolVal = 0.001

if(check_ortho(columnVector1, columnVector2, tolVal)):
    print("These vectors are orthogonal.\n")
else:
    print("These vectors are not orthogonal.\n")


################
#### PART 3 ####
################

print("Part 3\n\n")

# Part 3 (2pt)
# - Read the data set A.csv as a matrix A ∈ R^(30 * 6) via:
# > from numpy import genfromtxt

A = np.genfromtxt("A.csv", delimiter=',')

# 1. Compute the SVD of A...
tup1, tup2, tup3 = np.linalg.svd(np.array(A))


# ...And find:
# 2. The third right singular vector
print("THIRD RIGHT SINGULAR VECTOR: ")
print(tup1[-3])
print("\n\n")

# 3. The second singular value
print("SECOND SINGULAR VALUE: ")
print(tup2[1])
print("\n\n")

# 4. The fourth left singular vector
print("FOURTH LEFT SINGULAR VECTOR: ")
print(tup1[3])
print("\n\n")

# 5. The rank of A
print("Rank of matrix: ")
print(np.linalg.matrix_rank(np.array(A)))