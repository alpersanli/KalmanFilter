## KALMAN FILTER (GEO)

## Libraries
import numpy as np
import random


## Baslangic Sartlari
def baslangicSartlarinaDon():       # baslangicSartlarinaDon() komutu ile baslangic sartlarina doner
    # Parametreleri ilk degerlerine donduren fonksiyon

    # Teori Listeleri
    X = []
    Y = []
    Z = []
    U = []
    V = []
    W = []
    R = []
    S = []

    x = 10000000
    y = 20000000
    z = 26925824.03567252
    u = 1000
    v = 1000
    w = 2724.866969229874
    r = 35000000
    s = 3070

    X.append(x)
    Y.append(y)
    Z.append(z)
    R.append(r)
    U.append(u)
    V.append(v)
    W.append(w)
    S.append(s)

    yy = 6.6742 * (10 ** (-11))
    M = 5.97214 * (10 ** (24))
    mu = yy * M

    deltaT = 1
    AS = 3  # Adim sayisi

    Gamax = 10
    Gamay = 10
    Gamaz = 15
    Gamau = 0.02
    Gamav = 0.02
    Gamaw = 0.02

    # Olcum Listeleri
    X1 = []
    Y1 = []
    Z1 = []
    R1 = []
    U1 = []
    V1 = []
    W1 = []
    S1 = []
    olcum = []

    return X,Y,Z,U,V,W,R,S,x,y,z,u,v,w,r,s,yy,M,mu,deltaT,AS,Gamax,Gamay,Gamaz,Gamau,Gamav,Gamaw,X1,Y1,Z1,R1,U1,V1,W1,S1,olcum


X,Y,Z,U,V,W,R,S,x,y,z,u,v,w,r,s,yy,M,mu,deltaT,AS,Gamax,Gamay,Gamaz,Gamau,Gamav,Gamaw,X1,Y1,Z1,R1,U1,V1,W1,S1,olcum = baslangicSartlarinaDon()


## Teori
for i in range(AS) :     # 2 adim yapti aslinda bu range (5,7) ile ayni aralarindaki farka bak

    X.append(X[-1]+ deltaT * U[-1])      #[-1] Her zaman son degeri isleme alir
    Y.append(Y[-1] + deltaT * V[-1])
    Z.append(Z[-1] + deltaT * W[-1])
    R.append(((X[-1] ** (2)) + (Y[-1] ** (2)) + (Z[-1] ** (2))) ** (1 / 2))

    #r_vector = np.matrix([[X(-1) Y(-1) X(-1) X(-1) X(-1)])

    U.append(U[-1] + deltaT * (((-mu) * X[-1]) / ((R[-1]) ** 3)))
    V.append(V[-1] + deltaT * (((-mu) * Y[-1]) / ((R[-1]) ** 3)))
    W.append(W[-1] + deltaT * (((-mu) * Z[-1]) / ((R[-1]) ** 3)))
    S.append(((U[-1] ** (2)) + (V[-1] ** (2)) + (W[-1] ** (2))) ** (1 / 2))


X,Y,Z,U,V,W,R,S,x,y,z,u,v,w,r,s,yy,M,mu,deltaT,AS,Gamax,Gamay,Gamaz,Gamau,Gamav,Gamaw,X1,Y1,Z1,R1,U1,V1,W1,S1,olcum = baslangicSartlarinaDon()


## Olcum
for i in range(AS) :

    X1.append(X[-1] + Gamax * random.random())
    Y1.append(Y[-1] + Gamay * random.random())
    Z1.append(Z[-1] + Gamaz * random.random())
    R1.append(((X1[-1] ** (2)) + (Y1[-1] ** (2)) + (Z1[-1] ** (2))) ** (1 / 2))

    U1.append(U[-1] + Gamau * random.random())
    V1.append(V[-1] + Gamav * random.random())
    W1.append(W[-1] + Gamaw * random.random())
    S1.append(((U1[-1] ** (2)) + (V1[-1] ** (2)) + (W1[-1] ** (2))) ** (1 / 2))

    olcum.append(([[X1[-1]], [Y1[-1]], [Z1[-1]], [U1[-1]], [V1[-1]], [W1[-1]]]))


## Kalman
k_x = []
k_y = []
k_z = []
k_r = []
k_u = []
k_v = []
k_w = []
k_s = []

k_x.append(X1[0])   # Olcum degerlerinin ilk eklendi
k_y.append(Y1[0])
k_z.append(Z1[0])
k_r.append(R1[0])
k_u.append(U1[0])
k_v.append(V1[0])
k_w.append(W1[0])
k_s.append(S1[0])

k_x_prime = []
k_y_prime = []
k_z_prime = []
k_r_prime = []
k_u_prime = []
k_v_prime = []
k_w_prime = []
k_s_prime = []

prime = []
z = []
k_guncel = []
Normalize_inovasyon = []
k_r_vec = []
k_r_boy = []
k_v_boy = []

# P MATRISI
# P(0), 6x6’ lik bir matris, diagonal 10, diger degerler sifir verilecek:
P = np.identity(6) * 10
# print("\nP Matrix : \n", P)
# Q MATRISI
# Q matrisi ise 6x6’lİk ve diagonal degerleri 0.001 digerleri sifir plan bir matris:
Q = np.identity(6) * 0.001
# print("\nQ Matrix : \n", Q)
# G MATRISI
G = np.identity(6)
# print("\nG Matrix : \n", G)
# G Transpose
Gt = G.transpose()
# print("\nG Transpose Matrix : \n", Gt)
# H MATRISI
# H olcum birim matrisi 6x6
H = np.identity(6)
# print("\nH Matrix : \n", H)
# H Transpose
Ht = H.transpose()
# print("\nH Transpose Matrix : \n", Ht)
# R MATRISI
# R olcme simulasyonundaki random, disaridan gelen bilgiler
R = np.matrix([[Gamax ** 2, 0, 0, 0, 0, 0],
               [0, Gamay ** 2, 0, 0, 0, 0],
               [0, 0, Gamaz ** 2, 0, 0, 0],
               [0, 0, 0, Gamau ** 2, 0, 0],
               [0, 0, 0, 0, Gamav ** 2, 0],
               [0, 0, 0, 0, 0, Gamaw ** 2]])
# print("\nR Matrix : \n", R)

for i in range(AS) :


    A41 = -(deltaT * mu * (- 2 * k_x[-1] ** 2 + k_y[-1] ** 2 + k_z[-1] ** 2)) / (k_x[-1] ** 2 + k_y[-1] ** 2 + k_z[-1] ** 2) ** (5 / 2)
    A42 = (3 * deltaT * mu * k_x[-1] * k_y[-1]) / (k_x[-1] ** 2 + k_y[-1] ** 2 + k_z[-1] ** 2) ** (5 / 2)
    A43 = (3 * deltaT * mu * k_x[-1] * k_z[-1]) / (k_x[-1] ** 2 + k_y[-1] ** 2 + k_z[-1] ** 2) ** (5 / 2)
    A51 = (3 * deltaT * mu * k_x[-1] * k_y[-1]) / (k_x[-1] ** 2 + k_y[-1] ** 2 + k_z[-1] ** 2) ** (5 / 2)
    A52 = -(deltaT * mu * (k_x[-1] ** 2 - 2 * k_y[-1] ** 2 + k_z[-1] ** 2)) / (k_x[-1] ** 2 + k_y[-1] ** 2 + k_z[-1] ** 2) ** (5 / 2)
    A53 = (3 * deltaT * mu * k_y[-1] * k_z[-1]) / (k_x[-1] ** 2 + k_y[-1] ** 2 + k_z[-1] ** 2) ** (5 / 2)
    A61 = (3 * deltaT * mu * k_x[-1] * k_z[-1]) / (k_x[-1] ** 2 + k_y[-1] ** 2 + k_z[-1] ** 2) ** (5 / 2)
    A62 = (3 * deltaT * mu * k_y[-1] * k_z[-1]) / (k_x[-1] ** 2 + k_y[-1] ** 2 + k_z[-1] ** 2) ** (5 / 2)
    A63 = -(deltaT * mu * ((k_x[-1] ** (2)) + (k_y[-1] ** (2)) - 2 * (k_z[-1] ** 2))) / (k_x[-1] ** 2 + k_y[-1] ** 2 + k_z[-1] ** 2) ** (5 / 2)

    """
    # Bunu karsilastir
    print(A41)
    print(A42)
    print(A43)
    print(A51)
    print(A52)
    print(A53)
    print(A61)
    print(A62)
    print(A63)
    """

    # Jacobian Matrix
    Jacobian = np.matrix([
    [1, 0, 0, deltaT, 0, 0],
    [0, 1, 0, 0, deltaT, 0],
    [0, 0, 1, 0, 0, deltaT],
    [A41, A42, A43, 1, 0, 0],
    [A51, A52, A53, 0, 1, 0],
    [A61, A62, A63, 0, 0, 1]
    ])

    #print(Jacobian)

    # Kalman icin kepler denklemi
    k_x_prime.append(k_x[-1] + deltaT + k_u[-1])
    k_y_prime.append(k_y[-1] + deltaT + k_v[-1])
    k_z_prime.append(k_z[-1] + deltaT + k_w[-1])
    k_u_prime.append(k_u[-1] + deltaT * (((-mu) * k_x[-1]) / ((k_r[-1]) ** 3)))
    k_v_prime.append(k_v[-1] + deltaT * (((-mu) * k_y[-1]) / ((k_r[-1]) ** 3)))
    k_w_prime.append(k_w[-1] + deltaT * (((-mu) * k_z[-1]) / ((k_r[-1]) ** 3)))

    prime.append(([[k_x_prime], [k_y_prime], [k_z_prime], [k_u_prime], [k_v_prime], [k_w_prime]]))

    # Matrix islemleri
    P_Tahmin = ((Jacobian) * (P) * (Jacobian.transpose()))+((G)*(Q)*(G.transpose()))
    S = (H * P_Tahmin * H.transpose() + R)
    K = P_Tahmin * H.transpose() * (S**-1)

    # Inovasyon Surec
    Z.append(np.subtract(olcum[i] , prime[-1]))
    # Kestirim Denklemi
    k_guncel.append(prime[-1] + (K * Z[-1]))
print(k_guncel)
"""
    Normalize_inovasyon.append( (S ** (-1 / 2)) * Z[-1])

    # Guncel Kalman kestrim degerlerinin atanmasi
    k_x_prime[i + 1] = k_guncel[1]
    k_y_prime[i + 1] = k_guncel[2]
    k_z_prime[i + 1] = k_guncel[3]
    k_u_prime[i + 1] = k_guncel[4]
    k_v_prime[i + 1] = k_guncel[5]
    k_w_prime[i + 1] = k_guncel[6]

    k_r_vec.append([[k_x_prime[i+1]], [k_y_prime[i + 1]], [k_z_prime[i+1]], [k_u_prime[i+1]], [k_v_prime[i+1]], [k_w_prime[i+1]]])

    k_r_boy.append(((k_x_prime[i+1])**(2) + (k_y_prime[i+1])**(2) + (k_z_prime[i+1])**(2))**(1/2))
    k_v_boy.append(((k_u_prime[i+1])**(2) + (k_v_prime[i+1])**(2) + (k_w_prime[i+1])**(2))**(1/2))

    k_P_guncel = ((np.identity(6) - (K) * (H)) * P_Tahmin)
    P = k_P_guncel


"""



# MATRİSLERDE ÇARPMA İŞLEMİ .NP.DOT(A,B) GİBİ BİR ŞEY BU YANLIŞ MATLAB TEN YAZ DUZELTMESİ COK UZUN SURECEK
