#!/bin/env python

from numpy import *

#
# Function to calculate P-H keys
#

rottable3 = asarray(
    [36, 28, 25, 27, 10, 10, 25, 27,
     29, 11, 24, 24, 37, 11, 26, 26,
     8, 8, 25, 27, 30, 38, 25, 27,
     9, 39, 24, 24, 9, 31, 26, 26,
     40, 24, 44, 32, 40, 6, 44, 6,
     25, 7, 33, 7, 41, 41, 45, 45,
     4, 42, 4, 46, 26, 42, 34, 46,
     43, 43, 47, 47, 5, 27, 5, 35,
     33, 35, 36, 28, 33, 35, 2, 2,
     32, 32, 29, 3, 34, 34, 37, 3,
     33, 35, 0, 0, 33, 35, 30, 38,
     32, 32, 1, 39, 34, 34, 1, 31,
     24, 42, 32, 46, 14, 42, 14, 46,
     43, 43, 47, 47, 25, 15, 33, 15,
     40, 12, 44, 12, 40, 26, 44, 34,
     13, 27, 13, 35, 41, 41, 45, 45,
     28, 41, 28, 22, 38, 43, 38, 22,
     42, 40, 23, 23, 29, 39, 29, 39,
     41, 36, 20, 36, 43, 30, 20, 30,
     37, 31, 37, 31, 42, 40, 21, 21,
     28, 18, 28, 45, 38, 18, 38, 47,
     19, 19, 46, 44, 29, 39, 29, 39,
     16, 36, 45, 36, 16, 30, 47, 30,
     37, 31, 37, 31, 17, 17, 46, 44,
     12, 4, 1, 3, 34, 34, 1, 3,
     5, 35, 0, 0, 13, 35, 2, 2,
     32, 32, 1, 3, 6, 14, 1, 3,
     33, 15, 0, 0, 33, 7, 2, 2,
     16, 0, 20, 8, 16, 30, 20, 30,
     1, 31, 9, 31, 17, 17, 21, 21,
     28, 18, 28, 22, 2, 18, 10, 22,
     19, 19, 23, 23, 29, 3, 29, 11,
     9, 11, 12, 4, 9, 11, 26, 26,
     8, 8, 5, 27, 10, 10, 13, 27,
     9, 11, 24, 24, 9, 11, 6, 14,
     8, 8, 25, 15, 10, 10, 25, 7,
     0, 18, 8, 22, 38, 18, 38, 22,
     19, 19, 23, 23, 1, 39, 9, 39,
     16, 36, 20, 36, 16, 2, 20, 10,
     37, 3, 37, 11, 17, 17, 21, 21,
     4, 17, 4, 46, 14, 19, 14, 46,
     18, 16, 47, 47, 5, 15, 5, 15,
     17, 12, 44, 12, 19, 6, 44, 6,
     13, 7, 13, 7, 18, 16, 45, 45,
     4, 42, 4, 21, 14, 42, 14, 23,
     43, 43, 22, 20, 5, 15, 5, 15,
     40, 12, 21, 12, 40, 6, 23, 6,
     13, 7, 13, 7, 41, 41, 22, 20], dtype=int32).reshape((48,8))

subpix3 = asarray([ 
    0, 7, 1, 6, 3, 4, 2, 5,
    7, 4, 6, 5, 0, 3, 1, 2,
    4, 3, 5, 2, 7, 0, 6, 1,
    3, 0, 2, 1, 4, 7, 5, 6,
    1, 0, 6, 7, 2, 3, 5, 4,
    0, 3, 7, 4, 1, 2, 6, 5,
    3, 2, 4, 5, 0, 1, 7, 6,
    2, 1, 5, 6, 3, 0, 4, 7,
    6, 1, 7, 0, 5, 2, 4, 3,
    1, 2, 0, 3, 6, 5, 7, 4,
    2, 5, 3, 4, 1, 6, 0, 7,
    5, 6, 4, 7, 2, 1, 3, 0,
    7, 6, 0, 1, 4, 5, 3, 2,
    6, 5, 1, 2, 7, 4, 0, 3,
    5, 4, 2, 3, 6, 7, 1, 0,
    4, 7, 3, 0, 5, 6, 2, 1,
    6, 7, 5, 4, 1, 0, 2, 3,
    7, 0, 4, 3, 6, 1, 5, 2,
    0, 1, 3, 2, 7, 6, 4, 5,
    1, 6, 2, 5, 0, 7, 3, 4,
    2, 3, 1, 0, 5, 4, 6, 7,
    3, 4, 0, 7, 2, 5, 1, 6,
    4, 5, 7, 6, 3, 2, 0, 1,
    5, 2, 6, 1, 4, 3, 7, 0,
    7, 0, 6, 1, 4, 3, 5, 2,
    0, 3, 1, 2, 7, 4, 6, 5,
    3, 4, 2, 5, 0, 7, 1, 6,
    4, 7, 5, 6, 3, 0, 2, 1,
    6, 7, 1, 0, 5, 4, 2, 3,
    7, 4, 0, 3, 6, 5, 1, 2,
    4, 5, 3, 2, 7, 6, 0, 1,
    5, 6, 2, 1, 4, 7, 3, 0,
    1, 6, 0, 7, 2, 5, 3, 4,
    6, 5, 7, 4, 1, 2, 0, 3,
    5, 2, 4, 3, 6, 1, 7, 0,
    2, 1, 3, 0, 5, 6, 4, 7,
    0, 1, 7, 6, 3, 2, 4, 5,
    1, 2, 6, 5, 0, 3, 7, 4,
    2, 3, 5, 4, 1, 0, 6, 7,
    3, 0, 4, 7, 2, 1, 5, 6,
    1, 0, 2, 3, 6, 7, 5, 4,
    0, 7, 3, 4, 1, 6, 2, 5,
    7, 6, 4, 5, 0, 1, 3, 2,
    6, 1, 5, 2, 7, 0, 4, 3,
    5, 4, 6, 7, 2, 3, 1, 0,
    4, 3, 7, 0, 5, 2, 6, 1,
    3, 2, 0, 1, 4, 5, 7, 6,
    2, 5, 1, 6, 3, 4, 0, 7], dtype=int32).reshape((48,8))

def peano_hilbert_keys(ix, iy, iz, bits):

    x = asarray(ix, dtype=int)
    y = asarray(iy, dtype=int)
    z = asarray(iz, dtype=int)

    key      = zeros(x.shape, dtype=int64)
    rotation = 0
    
    for i in range(bits-1, -1, -1):
        mask = (1 << i)
        pix = where((x & mask), 4, 0) | where((y & mask), 2, 0) | where((z & mask), 1, 0)
        key = key << 3
        key = key | subpix3[rotation, pix]
        rotation = rottable3[rotation, pix]

    return key

#
# Function to calculate coords from keys
#

quadrants = asarray([
  0, 7, 1, 6, 3, 4, 2, 5,
  7, 4, 6, 5, 0, 3, 1, 2,
  4, 3, 5, 2, 7, 0, 6, 1,
  3, 0, 2, 1, 4, 7, 5, 6,
  1, 0, 6, 7, 2, 3, 5, 4,
  0, 3, 7, 4, 1, 2, 6, 5,
  3, 2, 4, 5, 0, 1, 7, 6,
  2, 1, 5, 6, 3, 0, 4, 7,
  6, 1, 7, 0, 5, 2, 4, 3,
  1, 2, 0, 3, 6, 5, 7, 4,
  2, 5, 3, 4, 1, 6, 0, 7,
  5, 6, 4, 7, 2, 1, 3, 0,
  7, 6, 0, 1, 4, 5, 3, 2,
  6, 5, 1, 2, 7, 4, 0, 3,
  5, 4, 2, 3, 6, 7, 1, 0,
  4, 7, 3, 0, 5, 6, 2, 1,
  6, 7, 5, 4, 1, 0, 2, 3,
  7, 0, 4, 3, 6, 1, 5, 2,
  0, 1, 3, 2, 7, 6, 4, 5,
  1, 6, 2, 5, 0, 7, 3, 4,
  2, 3, 1, 0, 5, 4, 6, 7,
  3, 4, 0, 7, 2, 5, 1, 6,
  4, 5, 7, 6, 3, 2, 0, 1,
  5, 2, 6, 1, 4, 3, 7, 0], dtype=int32).reshape((24,2,2,2))

rotxmap_table = asarray([4, 5, 6, 7, 8, 9, 10, 11,
                         12, 13, 14, 15, 0, 1, 2,
                         3, 17, 18, 19, 16, 23, 20,
                         21, 22], dtype=int32)

rotymap_table = asarray([1, 2, 3, 0, 16, 17, 18, 19,
                         11, 8, 9, 10, 22, 23, 20,
                         21, 14, 15, 12, 13, 4, 5,
                         6, 7], dtype=int32)

rotx_table = asarray([3, 0, 0, 2, 2, 0, 0, 1], dtype=int32)
roty_table = asarray([0, 1, 1, 2, 2, 3, 3, 0], dtype=int32)
sense_table = asarray([-1, -1, -1, +1, +1, -1, -1, -1], dtype=int32)

quadrants_inverse_x = ndarray((24,8), dtype=int32)
quadrants_inverse_y = ndarray((24,8), dtype=int32)
quadrants_inverse_z = ndarray((24,8), dtype=int32)

for rotation in range(24):
    for bitx in range(2):
        for bity in range(2):
            for bitz in range(2):
                quad = quadrants[rotation,bitx,bity,bitz]
		quadrants_inverse_x[rotation,quad] = bitx
		quadrants_inverse_y[rotation,quad] = bity
		quadrants_inverse_z[rotation,quad] = bitz

def peano_hilbert_key_inverses(key, bits):

    key = asarray(key, dtype=int)

    shift = 3 * (bits - 1)
    mask = 7 << shift

    rotation = zeros(key.shape, dtype=int32)
    sense = ones(key.shape, dtype=int32)

    x = zeros(key.shape, dtype=int32)
    y = zeros(key.shape, dtype=int32)
    z = zeros(key.shape, dtype=int32)

    for i in range(bits):
        
        keypart = (key & mask) >> shift

        quad = where(sense==1, keypart, 7-keypart)
        
        x = (x << 1) + quadrants_inverse_x[rotation,quad]
        y = (y << 1) + quadrants_inverse_y[rotation,quad]
        z = (z << 1) + quadrants_inverse_z[rotation,quad]

        rotx = rotx_table[quad]
        roty = roty_table[quad]
        sense *= sense_table[quad]

        if len(key.shape)>0:
            while(any(rotx>0)):
                ind = rotx>0
                rotation[ind] = rotxmap_table[rotation[ind]]
                rotx[ind] -= 1
            while(any(roty>0)):
                ind = roty>0
                rotation[ind] = rotymap_table[rotation[ind]]
                roty[ind] -= 1
        else:
            while(rotx>0):
                rotation = rotxmap_table[rotation]
                rotx -= 1
            while(roty>0):
                rotation = rotymap_table[rotation]
                roty -= 1

        mask >>= 3
        shift -= 3

    return x,y,z
