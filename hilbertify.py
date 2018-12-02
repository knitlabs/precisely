import numpy
import imageio


def imageToMatrix(path):
    return imageio.imread(path)

def rotate(n,x,y,rx,ry):
    if ry == 0:
        if rx == 1:
            x = n-1-x
            y = n-1-y
    return y,x

def xy2d(n,x,y):
    rx=0
    ry=0
    s= n//2
    d=0
    while(s > 0):
        rx =(x&s)>0
        ry =(y&s)>0
        d += s*s*((3*rx)^ry)
        x,y=rotate(s,x,y,rx,ry)
        s = s//2
        #rotate(s,x,y,rx,ry)
    return d


def unrollMatrixIntoHilbertCurve(imageMatrix,imageHeight,imageWidth):
    hilbertArray =[-1]*imageHeight*imageWidth
    k=0
    for i in range(0,imageHeight):
        for j in range(0,imageWidth):
            k = imageWidth - i -1
            pos = xy2d(imageWidth,j,k)
            hilbertArray[pos] = imageMatrix[i][j]
    return numpy.array(hilbertArray)

def imageToHilbert(path,imageHeight,imageWidth):
    imageMatrix=imageToMatrix(path)
    hilbertCurve=unrollMatrixIntoHilbertCurve(imageMatrix,imageHeight,imageWidth)
    userId=int(path.split('.')[0][-3:-1])
    return userId,meansVector(hilbertCurve,imageHeight,imageWidth)

def meansVector(unrolledImage,imageHeight,imageWidth):
    tonesNumber = 52
    size = imageWidth*imageHeight
    pxPerBlock = size//tonesNumber
    sum = 0
    pos = 0
    means = [-1]*52
    for i in range(size):
        pos = i%pxPerBlock
        if pos == 0 :
            sum = 0
        sum = sum + unrolledImage[i]
        if pos == (pxPerBlock - 1) :
            means[int(i/pxPerBlock)] = sum/pxPerBlock
    return numpy.array(means)
