import main as mn
import unittest
import numpy

class TestOperations(unittest.TestCase):

    def test_suma_v(self):
        suma1= mn.suma_vectores((3,2),(1,-1))
        self.assertAlmostEqual(suma1, 4 + 1j)

        suma2 = mn.suma_vectores((-5, -7), (0, +3))
        self.assertAlmostEqual(suma2, -5 -4j)

    def test_inverso(self):
        inverso1= mn.inverso((-1,-2))
        self.assertAlmostEqual(inverso1, 1 + 2j)

        inverso2 = mn.inverso((5, -5))
        self.assertAlmostEqual(inverso2, -5 +5j)

    def test_multiplicacion(self):
        multi1= mn.multiplicacion((1,1),(2))
        self.assertAlmostEqual(multi1, 2 + 2j)

        multi2 = mn.multiplicacion((3, -2),(3))
        self.assertAlmostEqual(multi2, 9 +-6j)


    def test_inverso_m(self):
        matriz1 = numpy.array([[1, 2], [3, 4]])


        matriz_inv1= mn.inverso_m(matriz1)


        matriz_test=numpy.linalg.inv(matriz1)


        numpy.testing.assert_almost_equal(matriz_inv1, matriz_test)

        matriz2 = numpy.array([[5, -6], [-7, 8]])


        matriz_inv2= mn.inverso_m(matriz2)


        matriz_test =numpy.linalg.inv(matriz2)


        numpy.testing.assert_almost_equal(matriz_inv2, matriz_test)

    def test_multiplicacion_esc(self):
        matriz1=numpy.array(([(2,2),(2,2)]))
        mult_esc1=mn.multiplicacion_esc((matriz1),(2))
        mult_test= 2*matriz1
        numpy.testing.assert_equal(mult_esc1, mult_test)

        matriz1=numpy.array(([(5,-2),(0,7)]))
        mult_esc1=mn.multiplicacion_esc((matriz1),(6))
        mult_test= 6*matriz1
        numpy.testing.assert_equal(mult_esc1, mult_test)

    def test_traspuesta(self):
        tras1 = mn.traspuesta_m(([(1,2),(3,4),(5,6)]))
        self.assertAlmostEqual(tras1, ([[1,3,5],[2,4,6]]))

        tras2 = mn.traspuesta_m(([(1,2),(3,4),(5,6)]))
        self.assertAlmostEqual(tras2, [[-3 + 1], [0 - 2]])


    def test_conjugada_m(self):
        matriz1=numpy.array([[1+2j],[3-4j],[5+6j]])
        conj1=mn.conjugada_m(matriz1)
        conj_test=numpy.conjugate(matriz1)
        numpy.testing.assert_equal(conj1, conj_test)

        matriz1=numpy.array([[-7+5j],[-3-9j],[1+1j]])
        conj1=mn.conjugada_m(matriz1)
        conj_test=numpy.conjugate(matriz1)
        numpy.testing.assert_equal(conj1, conj_test)


    def test_adjunta_m(self):
        matriz1=numpy.array(([[1+2j,3-4j],[5+6j,7-1j]]))
        adj1=mn.adjunta_m(matriz1)
        adj_test=numpy.conjugate(numpy.transpose(numpy.array(matriz1)))
        numpy.testing.assert_equal(adj1, adj_test)

        matriz1=numpy.array(([[9-2j,-3-4j],[-4+6j,1-0j]]))
        adj1=mn.adjunta_m(matriz1)
        adj_test=numpy.conjugate(numpy.transpose(numpy.array(matriz1)))
        numpy.testing.assert_equal(adj1, adj_test)
