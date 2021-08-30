"""
    Optimal power flow in power distribution grids using
    second-order cone optimization
    by:
        Alejandro Garces
        30/08/2021
        Version 01
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cvx

"----- Read the database -----"
feeder = pd.read_csv("FEEDER34.csv")
num_lines = len(feeder)
line = {}
load = {}
source = {}
num_nodes = 0
num_sources = 0
kk = 0
for k in range(num_lines):
    n1 = feeder['From'][k]
    n2 = feeder['To'][k]
    z  = feeder['Rpu'][k] + 1j*feeder['Xpu'][k]    
    line[n1,n2] = kk,1/z,feeder['SmaxLine'][k]
    kk = kk + 1
    num_nodes = np.max([num_nodes,n1+1,n2+1])
    load[n2] = feeder['Ppu'][k]+1j*feeder['Qpu'][k]
    if feeder['SGmax'][k] > 0.0:        
        source[n2] = (num_sources,feeder['PGmax'][k],feeder['SGmax'][k])
        num_sources = num_sources + 1

"----- Optimization model -----"
s_slack = cvx.Variable(complex=True)
u = cvx.Variable(num_nodes)
w = cvx.Variable(num_lines,complex=True)
s_from = cvx.Variable(num_lines,complex=True)
s_to   = cvx.Variable(num_lines,complex=True)
s_gen = cvx.Variable(num_sources,complex=True)
res = [u[0]==1.0]
EqN = num_nodes*[0]
for (k,m) in line:
    pos,ykm,smax = line[(k,m)]
    EqN[k] = EqN[k]-s_from[pos]
    EqN[m] = EqN[m]+s_to[pos]
    EqN[m] = EqN[m]-load[m]
for m in source:
    pos,pmax,smax = source[m]
    EqN[m] = EqN[m]+s_gen[pos]
    res += [cvx.abs(s_gen[pos]) <= smax]    
    res += [cvx.real(s_gen[pos]) <= pmax]
EqN[0] = EqN[0] + s_slack    
res += [EqN[0] == 0]    
for (k,m) in line:
    pos,ykm,smax = line[(k,m)]
    res += [cvx.SOC(u[k]+u[m],cvx.vstack([2*w[pos],u[k]-u[m]]))]
    res += [s_from[pos] == ykm.conjugate()*(u[k]-w[pos])]
    res += [s_to[pos] == ykm.conjugate()*(cvx.conj(w[pos])-u[m])]
    res += [u[m] >= 0.95**2]
    res += [u[m] <= 1.05**2]
    res += [cvx.abs(s_from[pos]) <= smax]
    res += [cvx.abs(s_to[pos]) <= smax]    
    res += [EqN[m] == 0]
res += [cvx.abs(s_slack)<=10]    
obj = cvx.Minimize(cvx.sum(cvx.real(s_from))-cvx.sum(cvx.real(s_to)))
OPFSOC = cvx.Problem(obj,res)
OPFSOC.solve(solver=cvx.ECOS,verbose=False)    
print(OPFSOC.status,obj.value)

"----- Print results -----"
v = np.sqrt(u.value)
a = np.zeros(num_nodes)
s = np.zeros(num_nodes)*0j
plim = np.zeros(num_nodes)
for (k,m) in line:
    pos,ykm,smax = line[(k,m)]    
    a[m] = a[k]-np.angle(w.value[pos])
for m in source:
    pos,pmax,smax = source[m]
    s[m] = s_gen[pos].value
    plim[m] = pmax
results = pd.DataFrame()
results['v(pu)'] = v
results['Ang(deg)'] = a*180/np.pi
results['Pgen'] = s.real
results['Pmax'] = plim
results['Qgen'] = s.imag
print(results.head(num_nodes))
Vn = v*np.exp(1j*a)


"----- Evaluate power loss -----"
Ybus = np.zeros((num_nodes,num_nodes))*1j
for b in line:
    k,m = b
    pos,ykm,smax = line[b]    
    Ybus[k,k] = Ybus[k,k] + ykm
    Ybus[k,m] = Ybus[k,m] - ykm
    Ybus[m,k] = Ybus[m,k] - ykm
    Ybus[m,m] = Ybus[m,m] + ykm
In = Ybus@Vn
Sloss = Vn.T@In.conjugate()
print(Sloss)
plt.plot(v)
plt.grid()
plt.xlabel('Nodes')
plt.ylabel('Voltages')
plt.show()
