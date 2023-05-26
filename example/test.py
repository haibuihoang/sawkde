import numpy as np 
import sawkde.backend
from sawkde import saw_gausian_kde 
import matplotlib.pyplot as plt


sample_size=1000
x_sample = np.random.normal(0, 5, sample_size)
dy = np.random.normal(0, 0.5, sample_size)
y_sample = x_sample/4+np.sin(x_sample)+dy


print("Plotting sample data...")
fig,ax=plt.subplots(1,1,figsize=(10,3))
ax.plot(x_sample,y_sample,'xb')
fig.savefig("data.png")


#Prepare grid to evalue KDE
xyvalues = np.vstack((x_sample,y_sample))
xx, yy = np.mgrid[-15:15:.1, -5:5:.1]
nx,ny=xx.shape[0],xx.shape[1]
pos = np.vstack((xx.ravel(), yy.ravel()))


#Default KDE (Scott bandwidth)
print("Plotting Fixed bandwidth KDE (Scott)...")
kde=saw_gausian_kde(xyvalues)
h_Scott=kde.factor
pdf_kde=kde.evaluate(pos).reshape(nx,ny)
fig,ax=plt.subplots(1,1,figsize=(10,3))
pl=ax.contourf(xx, yy, pdf_kde,cmap='gray_r')
fig.savefig("fw_kde.png")


#Default KDE with a half bw
print("Plotting Fixed bandwidth KDE with half bandwidth....")
kde=saw_gausian_kde(xyvalues,bw_method=h_Scott/2)
pdf_kde=kde.evaluate(pos).reshape(nx,ny)
fig,ax=plt.subplots(1,1,figsize=(10,3))
pl=ax.contourf(xx, yy, pdf_kde,cmap='gray_r')
fig.savefig("fw1_kde.png")



#Selective bandwidth KDE
print("Plotting Selective bandwidth KDE....")
h=np.array([h_Scott,h_Scott/2])
kde.set_selective_factor(h)
pdf_kde=kde.evaluate(pos).reshape(nx,ny)
fig,ax=plt.subplots(1,1,figsize=(10,3))
pl=ax.contourf(xx, yy, pdf_kde,cmap='gray_r')
fig.savefig("sw_kde.png")



#Selective bandwidth KDE
print("Plotting Selective-Adaptive bandwidth KDE....")
kde.calc_local_weights()
pdf_kde=kde.aw_evaluate(pos).reshape(nx,ny)
fig,ax=plt.subplots(1,1,figsize=(10,3))
pl=ax.contourf(xx, yy, pdf_kde,cmap='gray_r')
fig.savefig("saw_kde.png")





