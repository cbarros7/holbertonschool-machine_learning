#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = [(matrix[position][2:4])
              for position in range(len(matrix))]
# your code here
# for x in range(len(matrix)):
#    the_middle.append(matrix[x][2:4])
print("The middle columns of the matrix are: {}".format(the_middle))
