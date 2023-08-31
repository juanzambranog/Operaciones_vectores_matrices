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
