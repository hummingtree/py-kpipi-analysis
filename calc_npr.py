#!/usr/bin/python

import sys
import numpy as np
from scipy.integrate import quad
from scipy.misc import factorial
from itertools import product

import re

from copy import deepcopy

from scipy.interpolate import interp1d

def round_sig(x, nsig):
  return round(x, nsig-int(np.floor(np.log10(abs(x))))-1)

def format_number_with_err(cv, err):
  nord = int(np.floor(np.log10(abs(cv))))
  nsig = 1-int(np.floor(np.log10(abs(err))))
  tmp_err = err*10.0**nsig
  nsig += nord + 1
  cv_str = str(round_sig(cv,nsig))
  cv_str_len = len(cv_str)
  if nord > -1:
    cv_str_len -= 1
  else:
    cv_str_len -= abs(nord) + 1
  for i in range(0,nsig-cv_str_len):
    cv_str += "0"
#  return str("${0:s}({1:d})$ & ${2:.2f}$".format(cv_str,int(round_sig(tmp_err,2)),err/abs(cv)*100.0))
  return str("{0:s}({1:d})".format(cv_str,int(round_sig(tmp_err,2))))

def format_number_with_err_err(err_ref, err):
    nsig = 1-int(np.floor(np.log10(abs(err_ref))))
    tmp_err = err*10.0**nsig
    return str("({0:d})".format( int(tmp_err) ))

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
                    Z24qq = deepcopy(Z24)
                    dZ24qq = deepcopy(dZ24)
                else:
                    Z24gg = deepcopy(Z24)
                    dZ24gg = deepcopy(dZ24)
#                print scheme_count
#                print [ Z24gg, dZ24gg, Z24qq, dZ24qq ]
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

def multiplication( sZ24l, sZ24h, sZ32l, sZ32h, sZ24ID, jMQx, jSqrtLL, scheme ):
    """
    See David Murphy's result for ZA
    """
    ZA = 0.73457
    
    """
    See Elaine Goode's thesis table 3.3 for the Wilson coefficients
    """
    tau = (+0.0014606)+(-0.00060408j)
    #Wilson coefficient MSbar 3 GeV
    wc = np.array( [ (+2.90342e-1)+(-3.97252e-3)*tau, (+4.70099e-5)+(-8.09555e-5)*tau, (-5.22390e-5)+(+3.26016e-4)*tau ] )
    wcRe = wc.real
#    print wcRe
    wcIm = wc.imag
#    print wcIm
   
    """
    See table VIII and IX in arxiv:1708.03552
    """
    if scheme=="qq":
        msb5rismom = np.array( [[0.99112, 0., 0.], [0., 1.00084, 0.00506], [0., 0.00599, 1.02921]] )
    else:
        msb5rismom = np.array( [[1.00414, 0., 0.], [0., 1.00084, 0.00506], [0., 0.01576, 1.08781]] )

    """
    See table II in arxiv:1708.03552 for lattice spacing
    """
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

def chiral_extrapolation( Z24l, dZ24l, Z24h, dZ24h, Z32l, dZ32l, Z32h, dZ32h, Z24ID, dZ24ID, MQxlat, MQxlat_jack, sqrtLL, sqrtLL_jack, scheme):
   
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
        
        [ tmpRe, tmpIm ] = multiplication( sZ24l, sZ24h, sZ32l, sZ32h, sZ24ID, MQxlat, sqrtLL, scheme )
        
        ARe =  ARe + tmpRe
        ARe2 = ARe2 + tmpRe*tmpRe
        AIm =  AIm + tmpIm
        AIm2 = AIm2 + tmpIm*tmpIm

    for j in range(len(MQxlat_jack)):
        [ tmpRe, tmpIm ] = multiplication( Z24l, Z24h, Z32l, Z32h, Z24ID, MQxlat_jack[j], sqrtLL_jack[j], scheme )
        ARe =  ARe + tmpRe
        ARe2 = ARe2 + tmpRe*tmpRe
        AIm =  AIm + tmpIm
        AIm2 = AIm2 + tmpIm*tmpIm
    
    [ central_tmpRe, central_tmpIm ] = multiplication( Z24l, Z24h, Z32l, Z32h, Z24ID, MQxlat, sqrtLL, scheme )
    
#    print Z24l
#    print Z24h
#    print Z32l
#    print Z32h
    
#    print np.dot( np.linalg.inv(sZ24h), sZ24l )
#    print np.dot( np.linalg.inv(sZ32h), sZ32l )
#    print Z24ID

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
filename = "../correlator_fits/results/fit_params"

GF       = 1.16637e-5
Vud      = 0.9743
Vus      = 0.2253
a24IDinv = 1.0083

ARe_scale = 1e-8
AIm_scale = 1e-13

ARe_dict = {}
dARe_dict = {}
AIm_dict = {}
dAIm_dict = {}

for parity in ["1", "0", "2"]:
    for scheme in ["qq", "gg"]:

        if parity=="1": ntw = 3
        else: 
            if parity=="2": ntw = 2
            else: ntw = 0

        const_term = GF / np.sqrt(2.) * Vud * Vus * np.sqrt(3./2.) * np.sqrt(2.**ntw) / np.sqrt(2.) * a24IDinv**3 
        
        [ Z24IDgg, dZ24IDgg, Z24IDqq, dZ24IDqq ] = read_hummingtree_rst_file("24x64x24ID/24x64x24ID_ZBK_x3.8474_mass0.0011.rst")
#        print Z24IDqq
#        print dZ24IDqq
#        print Z24IDgg
#        print dZ24IDgg

        ### g,g
        [ Z24l, dZ24l, Z24h, dZ24h ] = load_nicolas_Z("L_all/L_naive_chiral_24cubed_"+scheme+".out", 1.4363, 3.) # 24I
        [ Z32l, dZ32l, Z32h, dZ32h ] = load_nicolas_Z("L_all/L_naive_chiral_32cubed_"+scheme+".out", 1.4363, 3.) # 24I
        
        ### ntw = 3
        
        MQ1lat = np.genfromtxt(filename+"/M-Q1-"+parity+".dat")
        MQ1lat_jack = np.genfromtxt(filename+"/M-Q1-"+parity+"_jacks.dat")
        MQ7lat = np.genfromtxt(filename+"/M-Q7-"+parity+".dat")
        MQ7lat_jack = np.genfromtxt(filename+"/M-Q7-"+parity+"_jacks.dat")
        MQ8lat = np.genfromtxt(filename+"/M-Q8-"+parity+".dat")
        MQ8lat_jack = np.genfromtxt(filename+"/M-Q8-"+parity+"_jacks.dat")
        
        sqrtLL = np.genfromtxt(filename+"/sqrtLL-"+parity+".dat")
        sqrtLL_jack = np.genfromtxt(filename+"/sqrtLL-"+parity+"_jacks.dat")
        
        Mlat = np.array([ MQ1lat, MQ7lat, MQ8lat ])
        Mlat_jack = [ np.array([ MQ1lat_jack[j], MQ7lat_jack[j], MQ8lat_jack[j] ]) for j in range(len(MQ1lat_jack)) ]
        
        if scheme=="qq":
            [ wcRe, dwcRe, wcIm, dwcIm ] = chiral_extrapolation( Z24l, dZ24l, Z24h, dZ24h, Z32l, dZ32l, Z32h, dZ32h, Z24IDqq, dZ24IDqq, Mlat, Mlat_jack, sqrtLL, sqrtLL_jack, scheme )
        else:
            [ wcRe, dwcRe, wcIm, dwcIm ] = chiral_extrapolation( Z24l, dZ24l, Z24h, dZ24h, Z32l, dZ32l, Z32h, dZ32h, Z24IDgg, dZ24IDgg, Mlat, Mlat_jack, sqrtLL, sqrtLL_jack, scheme )
        
        ARe = wcRe * const_term
        dARe = dwcRe * const_term
        AIm = wcIm * const_term
        dAIm = dwcIm * const_term

        ARe_dict[parity, scheme]  = ARe
        dARe_dict[parity, scheme] = dARe
        AIm_dict[parity, scheme]  = AIm
        dAIm_dict[parity, scheme] = dAIm

        print "parity = %s, ntw = %d, scheme = %s :" % (parity, ntw, scheme)
        print "A_2 = %+.6e %.6e %+.6e %.6e" % (ARe, dARe, AIm, dAIm)
    
    dARe_npr = abs( ARe_dict[parity, "qq"] - ARe_dict[parity, "gg"] ) 
    dAIm_npr = abs( AIm_dict[parity, "qq"] - AIm_dict[parity, "gg"] ) 
    s1 = format_number_with_err(ARe_dict[parity, "qq"]/ARe_scale, dARe_dict[parity, "qq"]/ARe_scale)
    s2 = format_number_with_err_err(dARe_dict[parity, "qq"]/ARe_scale, dARe_npr/ARe_scale)
    s3 = format_number_with_err(AIm_dict[parity, "qq"]/AIm_scale, dAIm_dict[parity, "qq"]/AIm_scale)
    s4 = format_number_with_err_err(dAIm_dict[parity, "qq"]/AIm_scale, dAIm_npr/AIm_scale)
    print "$ %s_\\mathrm{stat.}%s_\\mathrm{NPR} $ & $ %s_\\mathrm{stat.}%s_\\mathrm{NPR} $" % ( s1, s2, s3, s4 )

print "Done."

def compute( x1, x2, y1, y2, X ):
    return (y2-y1)/(x2-x1)*(X-x1)+y1

mK    = 0.50425
dmK   = 0.00049

Epp1  = 0.5634 
dEpp1 = 0.0040
#Epp0  = 0.28221 
#dEpp0 = 0.00070
Epp2  = 0.4768 
dEpp2 = 0.0017

order = 2
sample_size = 100

eARe_dict  = {}
edARe_dict = {}
eAIm_dict  = {}
edAIm_dict = {}

for scheme in ["qq", "gg"]:
    
    eARe  = 0.
    eARe2 = 0.
    eAIm  = 0.
    eAIm2 = 0.
    
    for s in range(sample_size):
        smK = mK + np.random.normal(scale=dmK)
        sEpp1 = Epp1 + np.random.normal(scale=dEpp1)
        sEpp0 = Epp2 + np.random.normal(scale=dEpp2)

        sARe1 = ARe_dict["1", scheme] + np.random.normal(scale=dARe_dict["1", scheme])
        sARe0 = ARe_dict["2", scheme] + np.random.normal(scale=dARe_dict["2", scheme])
        sAIm1 = AIm_dict["1", scheme] + np.random.normal(scale=dAIm_dict["1", scheme])
        sAIm0 = AIm_dict["2", scheme] + np.random.normal(scale=dAIm_dict["2", scheme])
        
        seARe = compute(sEpp1**order, sEpp0**order, sARe1, sARe0, smK**order)
        seAIm = compute(sEpp1**order, sEpp0**order, sAIm1, sAIm0, smK**order)
    
        eARe += seARe
        eARe2 += seARe**2
        eAIm += seAIm
        eAIm2 += seAIm**2
    
    central_tmpRe = compute(Epp1**order, Epp2**order, ARe_dict["1", scheme], ARe_dict["2", scheme], mK**order)
    central_tmpIm = compute(Epp1**order, Epp2**order, AIm_dict["1", scheme], AIm_dict["2", scheme], mK**order)
    
    dARe = np.sqrt( eARe2/sample_size-eARe*eARe/(sample_size*sample_size) )
    dAIm = np.sqrt( eAIm2/sample_size-eAIm*eAIm/(sample_size*sample_size) )
    
    eARe_dict[scheme] = central_tmpRe
    eAIm_dict[scheme] = central_tmpIm
    edARe_dict[scheme] = dARe
    edAIm_dict[scheme] = dAIm
 
    # print "A_2 = %+.6e %.6e %+.6e %.6e" % ( central_tmpRe, dARe, central_tmpIm, dAIm ) 
    # print [ order, central_tmpRe, dARe, central_tmpIm, dAIm ]

edARe_npr = abs( eARe_dict["qq"] - eARe_dict["gg"] ) 
edAIm_npr = abs( eAIm_dict["qq"] - eAIm_dict["gg"] ) 
s1 = format_number_with_err(eARe_dict["qq"]/ARe_scale, edARe_dict["qq"]/ARe_scale)
s2 = format_number_with_err_err(edARe_dict["qq"]/ARe_scale, edARe_npr/ARe_scale)
s3 = format_number_with_err(eAIm_dict["qq"]/AIm_scale, edAIm_dict["qq"]/AIm_scale)
s4 = format_number_with_err_err(edAIm_dict["qq"]/AIm_scale, edAIm_npr/AIm_scale)
print "$ %s_\\mathrm{stat.}%s_\\mathrm{NPR} $ & $ %s_\\mathrm{stat.}%s_\\mathrm{NPR} $" % ( s1, s2, s3, s4 )

