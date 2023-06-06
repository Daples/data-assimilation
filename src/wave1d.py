#
# 1d shallow water model
#
# solves
# dh/dt + D du/dx = 0
# du/dt + g dh/dx + f*u = 0
#
# staggered discretiztation in space and central in time
#
# o -> o -> o -> o ->   # staggering
# L u  h u  h u  h  R   # element
# 0 1  2 3  4 5  6  7   # index in state vector
#
# m=1/2, 3/2, ...
#  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m]
# = u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
# m=1,2,3,...
#  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])
# = h[n,m] - 0.5 D dt/dx ( u[n,m+1/2] - u[n,m-1/2])

from question1 import question1
from question3 import question3
from question4 import question4
from question5 import question5
from question6 import question6
from question7 import question7
from question8 import question8
from question9 import question9
from question10 import question10,loopQ10

if __name__ == "__main__":
    # question1()
    # question3()
    # question4()
    # question5()
    # question6()
    # question7()
    # question8()
    question9()
    #question10()
    #loopQ10()
