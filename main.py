import numpy


def suma_vectores(c1,c2):
    suma_vec= complex(c1[0], c1[1]) + complex(c2[0], c2[1])

    return suma_vec

def inverso(c1):
    inv= complex(c1[0]*-1,c1[1]*-1)
    return inv

def multiplicacion(c1,a):
    multi= complex(c1[0]*a,c1[1]*a)
    return multi
def suma_matrices(m1,m2):


    suma_mat = [[0 for i in range(len(m1[0]))] for i in range(len(m1))]

    for i in range(len(m1)):
        for j in range(len(m1[0])):
            suma_mat[i][j] = m1[i][j] + m2[i][j]

    return suma_mat

def inverso_m(m1):
    inv_m= numpy.linalg.inv(m1)

    return inv_m

def multiplicacion_esc(m1,a):
    mult_esc= (numpy.array(m1))*a

    return mult_esc
def traspuesta_m(m1):
    trasp_m=numpy.transpose(numpy.array(m1))
    m_nor=numpy.array(m1)
    return m_nor,trasp_m

def conjugada_m(m1):

    conju_m=numpy.conjugate(numpy.array(m1))

    return conju_m

def adjunta_m(m1):
    adj_m=numpy.conjugate(numpy.transpose(numpy.array(m1)))
    return adj_m

def producto_m(m1,m2):
    m1=numpy.array(m1)
    m2=numpy.array(m2)
    produc=numpy.mult(m1,m2)
    return produc

def accion(m1,v1):
    m1 = numpy.array(m1)
    v1 = numpy.array(v1)

    acc = numpy.dot(m1, v1)

    return acc

def interno(m1,m2):
    m1 = numpy.array(m1)
    m2 = numpy.array(m2)

    inter = numpy.dot(m1, m2)

    return inter

def norma(v1):
    v1 = numpy.array(v1)
    norma = numpy.linalg.norm(v1)
    return norma

def distancia(v1,v2):
    v1 = numpy.array(v1)
    v2 = numpy.array(v2)

    dist = numpy.linalg.norm(v1 - v2)
    return dist


def produc_tensor(m1,m2):
    m1=numpy.array(m1)
    m2=numpy.array(m2)
    produc_t=numpy.tensordot(m1,m2,axes=0)
    return produc_t



print(suma_vectores((3,2),(1,-1)))
print(suma_vectores((-5,-7),(0,3)))

print(inverso((-1,-2)))
print(inverso((5,-5)))

print(multiplicacion((1,1),(2)))

print(suma_matrices(([(1,1),(2,2)]),([(1,1),(2,2)])))

print(inverso_m([[1, 2], [3, 4]]))
print(inverso_m([[-5, -6], [7, 8]]))

print(multiplicacion_esc(([(2,2),(2,2)]),(2)))

print(traspuesta_m(([(1,2),(3,4),(5,6)])))

print(conjugada_m(([[1+2j],[3-4j],[5+6j]])))

print(adjunta_m(([[1+2j,3-4j],[5+6j,7-1j]])))

print(producto_m([[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]))

print(accion([[1, 2], [3, 4]],[5, 6]))

print(interno([[1, 2], [3, 4]],[[5, 6],[7,8]]))

print(norma([1, 2, 3]))

print(distancia([1, 2, 3],[4,5,6]))

print(produc_tensor([[3+2j,5-1j],[0,12]],[[1,1+3j],[10+2j,6]]))
