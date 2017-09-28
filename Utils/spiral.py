import numpy as np

numbInClass = 200
numbClasses = 5 

def makeSpiral(spId):
    radius = np.linspace(2, 20, numbInClass)
    origTheta = 2 * np.pi / float(numbClasses)
    theta = np.linspace(spId * origTheta, (spId+4) * origTheta, numbInClass)
    theta += 0.2 * np.random.uniform(low = -1, high = 1, size = numbInClass) * np.linspace(1, 1.5, numbInClass)
    xPos = radius * np.cos(theta) 
    yPos = radius * np.sin(theta) 
    return np.c_[xPos, yPos], np.array([spId] * numbInClass)


