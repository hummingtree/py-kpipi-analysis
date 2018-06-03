#!/usr/bin/python

import sys
import numpy as np
from scipy.integrate import quad
from scipy.misc import factorial
from itertools import product

import re

from copy import deepcopy

from scipy.interpolate import interp1d

def load_nicolas_Z(filename, low, high):
    Zl = np.array( [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]] )
    dZl = np.array( [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]] )
    Zh = np.array( [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]] )
    dZh = np.array( [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]] )
    with open(filename) as f:
        lines = f.readlines()
        line = 0
        while line<len(lines):
            if "iop=" in lines[line]:
                p = re.compile('iop=(.*) jop=(.*) ---')
                m  = p.search(lines[line])
                i = np.int( m.group(1) )
                j = np.int( m.group(2) )
#                print [i,j]
                mu = []
                Z  = []
                dZ = []
                line += 1
                while line < len(lines) and "iop=" not in lines[line]:
                    if "ip =" in lines[line]:
                        p = re.compile(' p = (.*) GeV')
                        m = p.search(lines[line])
                        mu.append(np.float(m.group(1)))
                        p = re.compile('y = (.*) \+\/\- (.*)')
                        m = p.search(lines[line+4])
                        Z.append(np.float(m.group(1)))
                        dZ.append(np.float(m.group(2)))
                    line += 1
#                print Z
#                print dZ
              
                gv = interp1d(mu, Z)
                ge = interp1d(mu, dZ)
 
                if i<4:
                    Zl[i-1][j-1] = np.float( gv(low) )
                    Zh[i-1][j-1] = np.float( gv(high) )
                    dZl[i-1][j-1] = np.float( ge(low) )
                    dZh[i-1][j-1] = np.float( ge(high) )

                line -= 1
            line += 1

        return [ Zl, dZl, Zh, dZh ]

#        sigma = np.dot( Zh, np.linalg.inv(Zl) )
#        print sigma
#        A = np.dot( Zh, np.linalg.inv(Zl) )
#        B = np.dot( dZl, np.sqrt( np.linalg.inv(Zl)*np.linalg.inv(Zl) ) )
#        d1 = np.dot( np.sqrt(A*A), np.sqrt(B*B) )
#        d2 = np.dot(dZh, np.sqrt( Zl*Zl ) )
#        print d1
#        print d2
#        dsigma = np.sqrt(d1*d1 + d2*d2)
#        print dsigma

def Wilson5MSbar_MSbar5RISMOM_RISMOM5lat(Zl, dZl, Zh, dZh):
    tau = (+0.0014606)+(-0.00060408j)
    wc = np.array( [ (+2.90342e-1)+(-3.97252e-3)*tau, (+4.70099e-5)+(-8.09555e-5)*tau, (-5.22390e-5)+(+3.26016e-4)*tau ] )
    msb5rismom = np.array( [0.99112, 0., 0.], [0., 1.00084, 0.00506], [0., 0.00599, 1.02921] )

def collapse_notation(val):
    c = len(val)-1
    z = 0
    while c >= 0:
        if(val[c] == '.'): break
        z += 1
        c -= 1
    return z

def read_hummingtree_rst_file(filename):
    Z24 = np.array( [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]] )
    dZ24 = np.array( [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]] )
    Z24gg = np.array( [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]] )
    Z24qq = np.array( [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]] )
    dZ24gg = np.array( [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]] )
    dZ24qq = np.array( [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]] )
    with open(filename) as f:
        lines = f.readlines()
        line = 0
        scheme_count = 0
        while line<len(lines):
            if 'real Z_5x5_zero:' in lines[line]:
                p = re.compile('(.*)\((.*)\)')
                m = p.search(lines[line+1].split('&')[0])
                Z24[0][0] = np.float(m.group(1))
                dZ24[0][0] = np.float(m.group(2))/10.**collapse_notation(m.group(1))
                
                m = p.search(lines[line+2].split('&')[1])
                Z24[1][1] = np.float(m.group(1))
                dZ24[1][1] = np.float(m.group(2))/10.**collapse_notation(m.group(1))

                m = p.search(lines[line+2].split('&')[2])
                Z24[1][2] = np.float(m.group(1))
                dZ24[1][2] = np.float(m.group(2))/10.**collapse_notation(m.group(1))
                
                m = p.search(lines[line+3].split('&')[1])
                Z24[2][1] = np.float(m.group(1))
                dZ24[2][1] = np.float(m.group(2))/10.**collapse_notation(m.group(1))
                
                m = p.search(lines[line+3].split('&')[2])
                Z24[2][2] = np.float(m.group(1))
                dZ24[2][2] = np.float(m.group(2))/10.**collapse_notation(m.group(1))
                
                if scheme_count>0:
                    Z24qq = Z24
                    dZ24qq = dZ24
                else:
                    Z24gg = Z24
                    dZ24gg = dZ24

                scheme_count += 1
            
            line += 1
        
        return [ Z24gg, dZ24gg, Z24qq, dZ24qq ] 

def generate_random_matrix( central, std ):
    rtn = deepcopy(central)
    for [i,j] in [[1,1], [2,2], [2,3], [3,2], [3,3]]:
        rtn[i-1,j-1] += np.random.normal(scale=std[i-1,j-1])
    return rtn

def chiral_extrapolation_test( Z24l, dZ24l, Z24h, dZ24h, Z32l, dZ32l, Z32h, dZ32h ):
    a24 = 1./1.785
    a32 = 1./2.383
    a24sqr = a24*a24
    a32sqr = a32*a32
    sample_size = 1000
    ssigma = np.array( [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]] )
    ssigma2 = np.array( [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]] )
    for s in range(sample_size):
        sZ24l = generate_random_matrix( Z24l, dZ24l )
        sZ24h = generate_random_matrix( Z24h, dZ24h )
        sZ32l = generate_random_matrix( Z32l, dZ32l )
        sZ32h = generate_random_matrix( Z32h, dZ32h )
#        sigma_ = ( a24sqr*np.dot( np.linalg.inv(sZ32l), sZ32h ) - a32sqr*np.dot( np.linalg.inv(sZ24l), sZ24h ) )/( a24sqr - a32sqr )
        sigma_ = ( a24sqr*np.dot( np.linalg.inv(sZ32h), sZ32l ) - a32sqr*np.dot( np.linalg.inv(sZ24h), sZ24l ) )/( a24sqr - a32sqr )
#        print sZ24l, sZ24h, sZ32l, sZ32h, sigma
#        print sZ24l
#        print sZ24h
#        print np.dot( sZ24h, np.linalg.inv(sZ24l) )
#        print sZ32l
#        print sZ32h
#        print np.dot( sZ32h, np.linalg.inv(sZ32l) )
        ssigma = ssigma + sigma_
        ssigma2 = ssigma2 + sigma_*sigma_

    central_sigma=( a24sqr*np.dot( np.linalg.inv(Z32h), Z32l ) - a32sqr*np.dot( np.linalg.inv(Z24h), Z24l ) )/( a24sqr - a32sqr )
#    central_sigma=( a24sqr*np.dot( np.linalg.inv(Z32l), Z32h ) - a32sqr*np.dot( np.linalg.inv(Z24l), Z24h ) )/( a24sqr - a32sqr )
    return [ central_sigma, np.sqrt( ssigma2/sample_size-ssigma*ssigma/(sample_size*sample_size) ) ] 

def multiplication( sZ24l, sZ24h, sZ32l, sZ32h, sZ24ID, jMQx, jSqrtLL ):
   
#TODO Need to divide by ZV/A and put in the -2 and -1/2
#DID.    
    ZA = 0.73457

    tau = (+0.0014606)+(-0.00060408j)
    #Wilson coefficient MSbar 3 GeV
    wc = np.array( [ (+2.90342e-1)+(-3.97252e-3)*tau, (+4.70099e-5)+(-8.09555e-5)*tau, (-5.22390e-5)+(+3.26016e-4)*tau ] )
    wcRe = wc.real
#    print wcRe
    wcIm = wc.imag
#    print wcIm
### q,q
#    msb5rismom = np.array( [[0.99112, 0., 0.], [0., 1.00084, 0.00506], [0., 0.00599, 1.02921]] )
### g,g
    msb5rismom = np.array( [[1.00084, 0., 0.], [0., 1.00084, 0.00506], [0., 0.01576, 1.08781]] )

    a24 = 1./1.785
    a32 = 1./2.383
    a24sqr = a24*a24
    a32sqr = a32*a32
   
    # linear extrapolation
    sigma = ( a24sqr*np.dot( np.linalg.inv(sZ32h), sZ32l ) - a32sqr*np.dot( np.linalg.inv(sZ24h), sZ24l ) )/( a24sqr - a32sqr )
    tmp1 = np.dot( sigma, ZA*ZA*sZ24ID )
    tmp2 = np.dot( msb5rismom, tmp1 )
    # basis changing
    tmp2[1][2] *= -0.5
    tmp2[2][1] *= -2.
    
    tmpRe = np.dot( wcRe, tmp2)
    ARe = np.dot( tmpRe, jSqrtLL*jMQx )
    tmpIm = np.dot( wcIm, tmp2)
    AIm = np.dot( tmpIm, jSqrtLL*jMQx )
    
    return [ ARe, AIm ]

def chiral_extrapolation( Z24l, dZ24l, Z24h, dZ24h, Z32l, dZ32l, Z32h, dZ32h, Z24ID, dZ24ID, MQxlat, MQxlat_jack, sqrtLL, sqrtLL_jack ):
   
    sample_size = 1000
    
    MRe = np.array( [0.,0.,0.] )
    MRe2 = np.array( [0.,0.,0.] )
    MIm = np.array( [0.,0.,0.] )
    MIm2 = np.array( [0.,0.,0.] )

    ARe = 0.
    ARe2 = 0.
    AIm = 0.
    AIm2 = 0.

    for s in range(sample_size):
        sZ24l = generate_random_matrix( Z24l, dZ24l )
        sZ24h = generate_random_matrix( Z24h, dZ24h )
        sZ32l = generate_random_matrix( Z32l, dZ32l )
        sZ32h = generate_random_matrix( Z32h, dZ32h )
        sZ24ID= generate_random_matrix( Z24ID, dZ24ID )
        
        [ tmpRe, tmpIm ] = multiplication( sZ24l, sZ24h, sZ32l, sZ32h, sZ24ID, MQxlat, sqrtLL )
        
        ARe =  ARe + tmpRe
        ARe2 = ARe2 + tmpRe*tmpRe
        AIm =  AIm + tmpIm
        AIm2 = AIm2 + tmpIm*tmpIm

    for j in range(len(MQxlat_jack)):
        [ tmpRe, tmpIm ] = multiplication( Z24l, Z24h, Z32l, Z32h, Z24ID, MQxlat_jack[j], sqrtLL_jack[j] )
        ARe =  ARe + tmpRe
        ARe2 = ARe2 + tmpRe*tmpRe
        AIm =  AIm + tmpIm
        AIm2 = AIm2 + tmpIm*tmpIm
    
    [ central_tmpRe, central_tmpIm ] = multiplication( Z24l, Z24h, Z32l, Z32h, Z24ID, MQxlat, sqrtLL )
    
    sample_size += len(MQxlat_jack)
    dARe = np.sqrt( ARe2/sample_size-ARe*ARe/(sample_size*sample_size) )
    dAIm = np.sqrt( AIm2/sample_size-AIm*AIm/(sample_size*sample_size) )
#    central_sigma=( a24sqr*np.dot( np.linalg.inv(Z32l), Z32h ) - a32sqr*np.dot( np.linalg.inv(Z24l), Z24h ) )/( a24sqr - a32sqr )
    
    return [ central_tmpRe, dARe, central_tmpIm, dAIm ]

#TODO Nicolas Garron's values are the inverse of the Z matrix we want, i.e. Z_V**2/Z_O

# Test to see if we could reproduce Nicolas' numbers.
[ Z24l, dZ24l, Z24h, dZ24h ] = load_nicolas_Z("L_all/L_naive_chiral_24cubed_qq.out", 3., 2.) # 24I
[ Z32l, dZ32l, Z32h, dZ32h ] = load_nicolas_Z("L_all/L_naive_chiral_32cubed_qq.out", 3., 2.) # 24I
[ sigma, dsigma ] = chiral_extrapolation_test( Z24l, dZ24l, Z24h, dZ24h, Z32l, dZ32l, Z32h, dZ32h )
#print sigma
#print dsigma

# Now the real calculation
[ Z24IDgg, dZ24IDgg, Z24IDqq, dZ24IDqq ] = read_hummingtree_rst_file("24x64x24ID/24x64x24ID_ZBK_x3.8474_mass0.0011.rst")
#print Z24IDqq
#print dZ24IDqq

### q,q
#[ Z24l, dZ24l, Z24h, dZ24h ] = load_nicolas_Z("L_all/L_naive_chiral_24cubed_qq.out", 1.4363, 3.) # 24I
#[ Z32l, dZ32l, Z32h, dZ32h ] = load_nicolas_Z("L_all/L_naive_chiral_32cubed_qq.out", 1.4363, 3.) # 24I

### g,g
[ Z24l, dZ24l, Z24h, dZ24h ] = load_nicolas_Z("L_all/L_naive_chiral_24cubed_gg.out", 1.4363, 3.) # 24I
[ Z32l, dZ32l, Z32h, dZ32h ] = load_nicolas_Z("L_all/L_naive_chiral_32cubed_gg.out", 1.4363, 3.) # 24I

### ntw = 3

filename = "../correlator_fits/results/fit_params"
parity   = "1"

MQ1lat = np.genfromtxt(filename+"/M-Q1-"+parity+".dat")
MQ1lat_jack = np.genfromtxt(filename+"/M-Q1-"+parity+"_jacks.dat")
MQ7lat = np.genfromtxt(filename+"/M-Q7-"+parity+".dat")
MQ7lat_jack = np.genfromtxt(filename+"/M-Q7-"+parity+"_jacks.dat")
MQ8lat = np.genfromtxt(filename+"/M-Q8-"+parity+".dat")
MQ8lat_jack = np.genfromtxt(filename+"/M-Q8-"+parity+"_jacks.dat")

sqrtLL = np.genfromtxt(filename+"/sqrtLL-"+parity+".dat")
sqrtLL_jack = np.genfromtxt(filename+"/sqrtLL-"+parity+"_jacks.dat")

#print MQ1lat
#print MQ1lat_jack

Mlat = np.array([ MQ1lat, MQ7lat, MQ8lat ])
Mlat_jack = [ np.array([ MQ1lat_jack[j], MQ7lat_jack[j], MQ8lat_jack[j] ]) for j in range(len(MQ1lat_jack)) ]

### q,q
#[ wcRe, dwcRe, wcIm, dwcIm ] = chiral_extrapolation( Z24l, dZ24l, Z24h, dZ24h, Z32l, dZ32l, Z32h, dZ32h, Z24IDqq, dZ24IDqq, Mlat, Mlat_jack, sqrtLL, sqrtLL_jack )
### g,g
[ wcRe, dwcRe, wcIm, dwcIm ] = chiral_extrapolation( Z24l, dZ24l, Z24h, dZ24h, Z32l, dZ32l, Z32h, dZ32h, Z24IDgg, dZ24IDgg, Mlat, Mlat_jack, sqrtLL, sqrtLL_jack )

GF  = 1.16637e-5
Vud = 0.9743
Vus = 0.2253
a24IDinv = 1.0083

const_term = GF / np.sqrt(2.) * Vud * Vus * np.sqrt(3./2.) * np.sqrt(2.**3) / np.sqrt(2.) * a24IDinv**3 

ARe = wcRe * const_term
dARe = dwcRe * const_term
AIm = wcIm * const_term
dAIm = dwcIm * const_term

print "A_2 = %+.6e +- %.6e %+.6e +- %.6e" % (ARe, dARe, AIm, dAIm)

### ntw = 0

parity   = "0"

MQ1lat = np.genfromtxt(filename+"/M-Q1-"+parity+".dat")
MQ1lat_jack = np.genfromtxt(filename+"/M-Q1-"+parity+"_jacks.dat")
MQ7lat = np.genfromtxt(filename+"/M-Q7-"+parity+".dat")
MQ7lat_jack = np.genfromtxt(filename+"/M-Q7-"+parity+"_jacks.dat")
MQ8lat = np.genfromtxt(filename+"/M-Q8-"+parity+".dat")
MQ8lat_jack = np.genfromtxt(filename+"/M-Q8-"+parity+"_jacks.dat")

sqrtLL = np.genfromtxt(filename+"/sqrtLL-"+parity+".dat")
sqrtLL_jack = np.genfromtxt(filename+"/sqrtLL-"+parity+"_jacks.dat")

#print MQ1lat
#print MQ1lat_jack

Mlat = np.array([ MQ1lat, MQ7lat, MQ8lat ])
Mlat_jack = [ np.array([ MQ1lat_jack[j], MQ7lat_jack[j], MQ8lat_jack[j] ]) for j in range(len(MQ1lat_jack)) ]

### q,q
#[ wcRe, dwcRe, wcIm, dwcIm ] = chiral_extrapolation( Z24l, dZ24l, Z24h, dZ24h, Z32l, dZ32l, Z32h, dZ32h, Z24IDqq, dZ24IDqq, Mlat, Mlat_jack, sqrtLL, sqrtLL_jack )
### g,g
[ wcRe, dwcRe, wcIm, dwcIm ] = chiral_extrapolation( Z24l, dZ24l, Z24h, dZ24h, Z32l, dZ32l, Z32h, dZ32h, Z24IDgg, dZ24IDgg, Mlat, Mlat_jack, sqrtLL, sqrtLL_jack )

const_term = GF / np.sqrt(2.) * Vud * Vus * np.sqrt(3./2.) * np.sqrt(2.**0) / np.sqrt(2.) * a24IDinv**3 

ARe = wcRe * const_term
dARe = dwcRe * const_term
AIm = wcIm * const_term
dAIm = dwcIm * const_term

print "A_2 = %+.6e +- %.6e %+.6e +- %.6e" % (ARe, dARe, AIm, dAIm)

#a = np.array([[3,1], [1,2]])
#b = np.array([[3,5], [1,2]])
#
#c = np.array([1,1])
#print np.sqrt(a)
#
#a[0][0] = 4.3
#
#print np.dot( c, b )
