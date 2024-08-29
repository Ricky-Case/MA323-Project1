import numpy as np
import matplotlib.pyplot as mplib

# (Hint: import numpy and matplotlib packages, and refer to quiver https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html)

# Visualize the addition of two input vectors "v1" and "v2" in 2D.
def vec_add_plot(v1, v2):
    print("Part 1\n")
    
    result = []
    
    if (len(v1) == 2 and len(v2) == 2):
        print("Computing ", v1, " + ", v2, "...")
        result = np.array(v1) + np.array(v2)
    else:
        print("Error: Please enter two vectors of length 2.")
    
    print("Result: ", result, "\n\n")
    
    # Visualize the vector addition with matplotlib.
    mplib.plot(v1, "r-", label = "Vector 1")
    mplib.plot(v2, "r-", label = "Vector 2")
    mplib.plot(result, "b-", label = "New Vector")
    
    mplib.title("Vector Addition")
    mplib.xlabel("X-Axis")
    mplib.ylabel("Y-Axis")
    
    # mplib.show()


# Check the orthogonality of two column vectors "v1" and "v2" of the same length, using tol as a scalar value strictly larger than 0.
def check_ortho(v1, v2, tol):
    print("Part 2\n")
    if (len(v1) != len(v2)):
        print("Error: Vector Dimension Mismatch!\n",
              "Please enter two vectors which have equivalent dimensions.")
    elif (tol <= 0):
        print("Error: Integer provided must be breater than 0.")
    else:
        cv1 = np.array(v1).T
        cv2 = np.array(v2).T
        
        print("Column Vector 1:\n", cv1, "\n")
        print("Column Vector 2:\n", cv2, "\n")
        
        # theta = np.arccos(np.dot(cv1, cv2)/(np.linalg.norm(cv1) * np.linalg.norm(cv2)))
        # checkValue = np.abs(np.pi/2 - theta)
        
        if((np.dot(cv1, cv2) < (0 - tol)) or (np.dot(cv1, cv2) > (0 + tol))):
            return False
        else:
            return True


################
#### PART 1 ####
################


vector1 = [4, 2]
vector2 = [-1, 2]
vec_add_plot(vector1, vector2)


################
#### PART 2 ####
################


columnVector1 = [5, 4, 4, 5, 1, 4, 1]
columnVector2 = [-1, -2, 1, -2, 3, 1, -5]
tolVal = 0.001

if(check_ortho(columnVector1, columnVector2, tolVal)):
    print("These vectors are orthogonal.\n")
else:
    print("These vectors are not orthogonal.\n")


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


# Part 3 (2pt)
# - Read the data set A.csv as a matrix A ∈ R^(30 * 6) via:
# > from numpy import genfromtxt

A = np.genfromtxt("A.csv", delimiter=',')

# 1. Compute the SVD of A...
svdA = np.linalg.svd(A)

s = np.linalg.svd([[1, 2, 3],
                   [2, 4, 6],
                   [-1, 1, -1]])


# ...And find:
# 2. The third right singular vector
# 3. The second singular value
# 4. The fourth left singular vector
# 5. The rank of A