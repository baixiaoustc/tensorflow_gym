__author__ = 'baixiao'
import matplotlib.pyplot as plt


eposide = [0,1,2,3,4]
steps = [12,22,33,44,34]
rewards = [45,60,70,80,0]


plt.figure(1)
plt.subplot(2, 1, 1)
plot1=plt.plot(eposide, steps, 'r')
plot2=plt.plot(eposide, rewards, 'g')
plt.subplot(2, 1, 2)
plot3=plt.plot(eposide, rewards, 'b')
plt.savefig("123.png")