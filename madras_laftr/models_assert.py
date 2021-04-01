import models
hls = {'enc':[10,7],'dec':[7, 10],'clas':[7,3],'adv':[7,3]}
xdim, ydim, adim, zdim = 13, 1, 1, 7
model = models.DemParGan(xdim, ydim, adim, zdim, hls)
#print(model.submodules) 
print(model.variables)


model = models.EqOddsUnweightedGan(xdim, ydim, adim, zdim, hls)
#print(model.submodules) 
print(model.variables)

model = models.EqOppUnweightedGan(xdim, ydim, adim, zdim, hls)
#print(model.submodules) 
print(model.variables)