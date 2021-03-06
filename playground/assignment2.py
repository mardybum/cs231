import numpy as np

x = np.zeros((3, 4, 3))
print(x.shape)

x[0] = [[88,10,1,], [1,2,1,], [1,2,1,], [1,2,1,]]
x[1] = [[1,10,1,], [1,2,1,], [1,2,1,], [1,2,1,]]
print(x.shape)

x[2] = [[1,10,1,], [1,2,1,], [1,2,1,], [1,2,1,]]
print(x.shape)
print(x[0, 0 , 0])

[[[[ 0.          0.          0.          0.          0.
     0.        ]
   [ 0.         -0.1        -0.09368421 -0.08736842 -0.08105263
     0.        ]
   [ 0.         -0.07473684 -0.06842105 -0.06210526 -0.05578947
     0.        ]
   [ 0.         -0.04947368 -0.04315789 -0.03684211 -0.03052632
     0.        ]
   [ 0.         -0.02421053 -0.01789474 -0.01157895 -0.00526316
     0.        ]
   [ 0.          0.          0.          0.          0.
     0.        ]]

  [[ 0.          0.          0.          0.          0.
     0.        ]
   [ 0.          0.00105263  0.00736842  0.01368421  0.02
     0.        ]
   [ 0.          0.02631579  0.03263158  0.03894737  0.04526316
     0.        ]
   [ 0.          0.05157895  0.05789474  0.06421053  0.07052632
     0.        ]
   [ 0.          0.07684211  0.08315789  0.08947368  0.09578947
     0.        ]
   [ 0.          0.          0.          0.          0.
     0.        ]]

  [[ 0.          0.          0.          0.          0.
     0.        ]
   [ 0.          0.10210526  0.10842105  0.11473684  0.12105263
     0.        ]
   [ 0.          0.12736842  0.13368421  0.14        0.14631579
     0.        ]
   [ 0.          0.15263158  0.15894737  0.16526316  0.17157895
     0.        ]
   [ 0.          0.17789474  0.18421053  0.19052632  0.19684211
     0.        ]
   [ 0.          0.          0.          0.          0.
     0.        ]]]


 [[[ 0.          0.          0.          0.          0.
     0.        ]
   [ 0.          0.20315789  0.20947368  0.21578947  0.22210526
     0.        ]
   [ 0.          0.22842105  0.23473684  0.24105263  0.24736842
     0.        ]
   [ 0.          0.25368421  0.26        0.26631579  0.27263158
     0.        ]
   [ 0.          0.27894737  0.28526316  0.29157895  0.29789474
     0.        ]
   [ 0.          0.          0.          0.          0.
     0.        ]]

  [[ 0.          0.          0.          0.          0.
     0.        ]
   [ 0.          0.30421053  0.31052632  0.31684211  0.32315789
     0.        ]
   [ 0.          0.32947368  0.33578947  0.34210526  0.34842105
     0.        ]
   [ 0.          0.35473684  0.36105263  0.36736842  0.37368421
     0.        ]
   [ 0.          0.38        0.38631579  0.39263158  0.39894737
     0.        ]
   [ 0.          0.          0.          0.          0.
     0.        ]]

  [[ 0.          0.          0.          0.          0.
     0.        ]
   [ 0.          0.40526316  0.41157895  0.41789474  0.42421053
     0.        ]
   [ 0.          0.43052632  0.43684211  0.44315789  0.44947368
     0.        ]
   [ 0.          0.45578947  0.46210526  0.46842105  0.47473684
     0.        ]
   [ 0.          0.48105263  0.48736842  0.49368421  0.5
     0.        ]
   [ 0.          0.          0.          0.          0.
     0.        ]]]]