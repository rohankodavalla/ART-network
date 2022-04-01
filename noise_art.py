#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Adaptive Resonance Theory
# Copyright (C) 2011 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
# Reference: Grossberg, S. (1987)
#            Competitive learning: From interactive activation to
#            adaptive resonance, Cognitive Science, 11, 23-63
#
# Requirements: python 2.5 or above => http://www.python.org 
#               numpy  1.0 or above => http://numpy.scipy.org
# -----------------------------------------------------------------------------
from __future__ import print_function
from __future__ import division
import numpy as np


class ART:
    ''' ART class

    Usage example:
    --------------
    # Create a ART network with input of size 5 and 20 internal units
    >>> network = ART(5,10,0.5)
    '''

    def __init__(self, n=5, m=10, rho=.5):
        '''
        Create network with specified shape

        Parameters:
        -----------
        n : int
            Size of input
        m : int
            Maximum number of internal units 
        rho : float
            Vigilance parameter
        '''
        # Comparison layer
        self.F1 = np.ones(n)
        # Recognition layer
        self.F2 = np.ones(m)
        # Feed-forward weights
        self.Wf = np.random.random((m,n))
        # Feed-back weights
        self.Wb = np.random.random((n,m))
        # Vigilance
        self.rho = rho
        # Number of active units in F2
        self.active = 0


    def learn(self, X):
        ''' Learn X '''

        # Compute F2 output and sort them (I)
        self.F2[...] = np.dot(self.Wf, X)
        I = np.argsort(self.F2[:self.active].ravel())[::-1]

        for i in I:
            # Check if nearest memory is above the vigilance level
            d = (self.Wb[:,i]*X).sum()/X.sum()
            if d >= self.rho:
                # Learn data
                self.Wb[:,i] *= X
                self.Wf[i,:] = self.Wb[:,i]/(0.5+self.Wb[:,i].sum())
                return self.Wb[:,i], i

        # No match found, increase the number of active units
        # and make the newly active unit to learn data
        if self.active < self.F2.size:
            i = self.active
            self.Wb[:,i] *= X
            self.Wf[i,:] = self.Wb[:,i]/(0.5+self.Wb[:,i].sum())
            self.active += 1
            return self.Wb[:,i], i

        return None,None


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    np.random.seed(1)

    # Example 1 : very simple data
    # -------------------------------------------------------------------------
    
           

   
    
    # Example 2 : Learning letters
    # -------------------------------------------------------------------------
    def letter_to_array(letter):
        ''' Convert a letter to a numpy array '''
        shape = len(letter), len(letter[0])
        Z = np.zeros(shape, dtype=int)
        for row in range(Z.shape[0]):
            for column in range(Z.shape[1]):
                if letter[row][column] == '#':
                    Z[row][column] = 1
        return Z

    def print_letter(Z):
        ''' Print an array as if it was a letter'''
        for row in range(Z.shape[0]):
            for col in range(Z.shape[1]):
                if Z[row,col]:
                    print( '#', end="" )
                else:
                    print( ' ', end="" )
            print( )

 


    A1=letter_to_array([' #    # ',
                        '#      #',
                        '       #',
                        '   #   #',
                        '#      #',
                        '#      #',
                        '        ',
                        '#      #'])
                      

    B1=letter_to_array(['#  #  # ',
                        '       #',
                        '       #',
                        '#   # # ',
                        '#      #',
                        '#      #',
                        '# ##  # ',
                        '        '])
    C1=letter_to_array([' #  # # ',
                        '#       ',
                        '        ',
                        '#       ',
                        '        ',
                        '#       ',
                        '#      #',
                        '  # # # '])
    D1=letter_to_array(['# #  ## ',
                        '#       ',
                        '       #',
                        '#      #',
                        '        ',
                        '#       ',
                        '#      #',
                        '# # # # '])
    E1=letter_to_array(['#  #  ##',
                        '#       ',
                        '#       ',
                        '#  # #  ',
                        '        ',
                        '#       ',
                        '#       ',
                        '##  #  #'])
    F1=letter_to_array(['#  #  ##',
                        '        ',
                        '#       ',
                        '# # # ##',
                        '        ',
                        '        ',
                        '#       ',
                        '#       '])
    G= letter_to_array([' #  # # ',
                        '#      #',
                        '        ',
                        '#       ',
                        '      ##',
                        '       #',
                        '#      #',
                        ' #  # # '])
    H= letter_to_array(['#      #',
                        '        ',
                        '#      #',
                        '#  #  # ',
                        '#      #',
                        '        ',
                        '       #',
                        '#      #'])
    I= letter_to_array(['# #    #',
                        '   #    ',
                        '   #    ',
                        '        ',
                        '        ',
                        '   #    ',
                        '   #    ',
                        '##  # ##'])
    J= letter_to_array(['##  #  #',
                        '       #',
                        '        ',
                        '       #',
                        '        ',
                        '       #',
                        '       #',
                        '#  ##   '])
    K= letter_to_array(['#    #  ',
                        '    #   ',
                        '# #     ',
                        '#       ',
                        '  #     ',
                        '#   #   ',
                        '        ',
                        '#     ##'])
    L= letter_to_array(['#       ',
                        '        ',
                        '#       ',
                        '#       ',
                        '        ',
                        '        ',
                        '#       ',
                        '#    # #'])
    M= letter_to_array(['#      #',
                        '        ',
                        '#  # #  ',
                        '       #',
                        '#       ',
                        '#      #',
                        '        ',
                        '#      #'])
    N= letter_to_array(['#      #',
                        ' #      ',
                        '       #',
                        '#  #    ',
                        '       #',
                        '#    #  ',
                        '        ',
                        '#      #'])
    O= letter_to_array(['##   #  ',
                        '#      #',
                        '#      #',
                        '       #',
                        '#       ',
                        '#      #',
                        '#       ',
                        '###   ##'])
    P= letter_to_array(['#  ###  ',
                        '#      #',
                        '#       ',
                        '#  #   #',
                        '#       ',
                        '        ',
                        '#       ',
                        '        '])
    Q= letter_to_array(['## # # #',
                        '##     #',
                        '#       ',
                        '#  #   #',
                        '    #  #',
                        '#       ',
                        '       #',
                        '#  #  ##'])
    R= letter_to_array(['#  # # #',
                        '#     # ',
                        '   #     ',
                        '# #      ',
                        '# #     ',
                        '#  #    ',
                        '      # ',
                        '#      #'])
    S= letter_to_array(['##  #  #',
                        '#       ',
                        '        ',
                        '#   #  #',
                        '       #',
                        '        ',
                        '       #',
                        '# # # ##'])
    T= letter_to_array(['# #   # ',
                        '   #    ',
                        '   #    ',
                        '        ',
                        '   #    ',
                        '        ',
                        '        ',
                        '   #    '])










    #samples = [A,B,C,D,E,F]
    #samples=[A,B_,C,D,E,F]
    samples=[A1,B1,C1,D1,E1,F1,G,H,I,J,K,L,M,N,O,P,Q,R,S,T]
    rho_value = 0.95
    #network = ART( 8*16, 64, rho=0.7 )
    network = ART( 8*8, 64, rho = rho_value)
    print("rho value is --> ",rho_value)
    print("Noise rate = 25%")
    for i in range(len(samples)):
        Z, k = network.learn(samples[i].ravel())
        print("Input character %c"%(ord('A')+i),"-> class recognized",chr(65+int(k)))
        #print_letter(Z.reshape(8,16))
        print_letter(Z.reshape(8,8))

