from models import DemParGan
hls = {'enc':[10,7],'dec':[10,7],'clas':[10,7],'adv':[10,7]}
model = DemParGan(14,2,1,7,hls)
print(model.submodules)
print(model.variables)