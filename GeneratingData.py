import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getUnitWeight(sandtype,DR):
    '''
    This functions calculates the gamma (unit weight) of soil.
    '''
    emax = 0.78
    emin = 0.48
    Gs = 2.65
    voidRatio = emax-(emax-emin)*DR/100
    density = Gs/(1 + voidRatio)
    unitWeight = density * 9.81
    
    return unitWeight

def calculateqc(DR,sigmah,phi_c):
    qc = 1.64 * (np.exp(0.1041 * phi_c + (0.0264-0.0002 * phi_c) * DR)) * ((sigmah / 100)**(0.841-0.0047 * DR) / 10)
    return qc

def getqc_z(sandtype,DR,L,B,depthIncr):
    '''
    This function calculates the qc given the z
    sandtype: ottawa sand
    DR: relative density
    L: Length of pile
    B: Outer diameter of pile
    depthIncr: .2
    '''
    K0 = 0.45
    phi_c = 30 #ottawa sand
    unitWeight = getUnitWeight(sandtype,DR)
    
    z = np.arange(start=depthIncr, stop = L + B, step = depthIncr)
    
    sigmah=z*unitWeight*K0
    
    qc = calculateqc(DR,sigmah,phi_c)
    
    qc_z = np.stack((z,qc))
    
    return qc_z

def plot(qc):
    plt.figure(figsize=(8, 6), dpi=80)
    plt.xlabel("qc") 
    plt.ylabel("z") 
    plt.gca().invert_yaxis()
    return plt.plot(qc[1],qc[0]) 

def getqc_z_nlayer(numOfLayer,sandtype,thickness,DR,L,B,depthIncr):
    '''
    This function calculates the qc wrt z for a soil with multiple layers
    '''
    K0 = 0.45
    phi_c = 30
    z = np.arange(start=depthIncr, stop = L + B, step = depthIncr)
    
    # Matrix instantiation
    unitWeight = np.zeros(shape = (len(z)))
    distributedDR = np.zeros(shape = (len(z)))
    boundryDepth = np.zeros(shape = (len(thickness)))
    boundryIndex = np.zeros(shape = (len(thickness)))
       
    for i in range(len(thickness)):
        boundryDepth[i] = np.sum(thickness[:i+1])
        boundryIndex[i] = round(boundryDepth[i]/depthIncr)
        
    boundryIndex = np.concatenate([[0], boundryIndex, [len(z)]])
    
    
    for i in range(len(boundryIndex)-1):
        unitWeight[int(boundryIndex[i]+1):int(boundryIndex[i+1]+1)] = getUnitWeight(sandtype[i],DR[i])
        distributedDR[int(boundryIndex[i]+1):int(boundryIndex[i+1]+1)] = DR[i]
    sigmaV = np.cumsum(unitWeight) * depthIncr
    sigmah = sigmaV * K0
    
    qc = calculateqc(distributedDR,sigmah,phi_c)
    qc_z = np.stack((z,qc))
    
    return qc_z

def getH_SingleLayer(Ip,L,sandtype,DR,h):
    '''
    This function determines the lateral capacity (H) of the pile in a single layer soil
    '''
    def getH(c1, c2, c3, c4, c5, a1, a2, a3, a4, b):
        Lcrit = (Ip ** 0.25) * (c1 * DR / 100 + c2 * np.log(Ip ** 0.25) + c3)
        betta = c4 * (Lcrit / Ip ** 0.25) + c5
        
        if L < Lcrit:
                ratio = (0.5 * np.sin(np.pi * L / Lcrit - np.pi / 2) + 0.5) ** betta
        else:
                ratio = 1
            
        Hcrit = 100 * (((a1 * h + a2) * DR / 100 + a3 * h + a4) * (Ip ** 0.72) + b)
        tempH = ratio * Hcrit
        
        return tempH
    
    if sandtype == 'Ottawa Sand':
        
        c1_05, c2_05, c3_05, c4_05, c5_05 = -12.5, -3, 37.9, 0.084, -1.2
        a1_05, a2_05, a3_05, a4_05, b_05 = -0.95, 49, -0.34, 25.80, -0.01
        H05 = getH(c1_05, c2_05, c3_05, c4_05, c5_05, a1_05, a2_05, a3_05, a4_05, b_05)
        
        c1_1, c2_1, c3_1, c4_1, c5_1 = -12, -3.3, 40.3, 0.08, -1.25
        a1_1, a2_1, a3_1, a4_1, b_1 = -2.05, 107, -0.35, 33.5, -0.07
        H1 = getH(c1_1, c2_1, c3_1, c4_1, c5_1, a1_1, a2_1, a3_1, a4_1, b_1)
        
        H= np.stack((H05,H1))
        
        
    else: #toyoura sand
        
        c1_05, c2_05, c3_05, c4_05, c5_05 = -12.5, -3, 37.9, 0.084, -1.2
        a1_05, a2_05, a3_05, a4_05, b_05 = -0.85, 32, -0.35, 33.5, -0.15
        H05 = getH(c1_05, c2_05, c3_05, c4_05, c5_05, a1_05, a2_05, a3_05, a4_05, b_05)
        
        c1_1, c2_1, c3_1, c4_1, c5_1 = -12, -3.3, 40.3, 0.08, -1.25
        a1_1, a2_1, a3_1, a4_1, b_1 = -1.30, 65, -0.42, 43, -0.02
        H1 = getH(c1_1, c2_1, c3_1, c4_1, c5_1, a1_1, a2_1, a3_1, a4_1, b_1)
        
        H = np.stack((H05,H1))    

    return H

def getH_TwoLayer(thickness,sandtype,DR,Ip,L,B,h):
    '''
    This function determines the lateral capacity (H) of the pile in a two-layer soil
    '''    
    Htop = getH_SingleLayer(Ip,L,sandtype[0],DR[0],h)
    Hbot = getH_SingleLayer(Ip,L,sandtype[1],DR[1],h)
    if L/B >= 5:
        alpha = -0.23
    else:
        alpha = 1/(-1.07*L/B+1)
        
        
    H2Layer_05 = Htop[0] + (1 - Htop[0] / Hbot[0]) * np.exp(alpha * (thickness / B) ** 1.5) * Hbot[0]
    H2Layer_1 = Htop[1] + (1 - Htop[1] / Hbot[1]) * np.exp(alpha * (thickness / B) ** 1.5) * Hbot[1]
    
    H = np.stack((H2Layer_05, H2Layer_1))
    
    return H

def getH_Threelayer(sandtype,DR,thickness,h,Ip,L,B):
    '''
    This function determines the lateral capacity (H) of the pile in a three-layer soil
    ''' 
    
    HAAA = getH_SingleLayer(Ip,L,sandtype[0],DR[0],h)
    
    HBAA = getH_TwoLayer(thickness[0], [sandtype[1], sandtype[0]],[DR[1], DR[0]],Ip,L,B,h)
    HBBC = getH_TwoLayer(thickness[0]+thickness[1], [sandtype[1], sandtype[2]],[DR[1], DR[2]],Ip,L,B,h)
    
    HABC_05 = HAAA[0] - HBAA[0] + HBBC[0]
    HABC_1 = HAAA[1] - HBAA[1] + HBBC[1]
    
    H = np.stack((HABC_05,HABC_1))
    
    return H

def getH05H1(B,L,tw,h,DR,sandType,numberOfLayers,thickness):
    '''
    This function determines the lateral capacity (H) of the pile
    '''
    ID = B-2*tw
    Ip = np.pi*(B**4-ID**4)/64
    if numberOfLayers == 1:
        H = getH_SingleLayer(Ip,L,sandType,DR,h)
    elif numberOfLayers == 2:
        H = getH_TwoLayer(thickness,sandType,DR,Ip,L,B,h)
    else:
        H = getH_Threelayer(sandType,DR,thickness,h,Ip,L,B)
    return H

# Generating data for three-layer soil

i = 1 #for file names
data_ThreeLayer = []
import random
depthIncr = 0.2 #increment size for depth z, unit:m 
numberOfLayers = 3
for i in range(1):
    B = 2 + np.random.random() * 8
    L = 3 * B + 12 * B * np.random.random()
    inp= np.zeros((5+len(np.arange(start = depthIncr, stop = L + B, step = depthIncr)),2))
    
    wallRatio = 40 + np.random.random() * 60
    tw= 1 / wallRatio * B
    ID = B - 2 * tw
    Ip = np.pi * (B**4 - ID**4) / 64
    h = 15 + 15 * random.random()
    t1 = 0.5 * B + (L - 0.5 * B) * random.random()
    t2 = (L - t1) * np.random.random()
    DR1 = 35 + 55 * np.random.random()
    DR2 = 35 + 55 * np.random.random()
    DR3 = 35 + 55 * np.random.random()
    DR = np.array([DR1, DR2, DR3])
    # t_1.append(t1)
    # t_2.append(t2)
    
    sandtype = ['Ottawa Sand', 'Ottawa Sand', 'Ottawa Sand']
    thickness = np.array([t1, t2])
    
    tempH = getH05H1(B,L,tw,h,DR,sandtype,numberOfLayers,thickness)
    
    inp[:5,:1] = np.concatenate([[tempH[0]], [tempH[1]], [L], [Ip], [h]]).reshape(5,1)
    
    qc_z = getqc_z_nlayer(numberOfLayers,sandtype,thickness,DR,L,B,depthIncr).T
    qc_z[:,1:] = qc_z[:,1:] + 0.5 * np.random.randn(qc_z.shape[0],1)
    
    inp[5:,:] = qc_z
    data_ThreeLayer.append(inp)
    i += 1
    pd.DataFrame(data_ThreeLayer[0]).to_csv('1.csv')