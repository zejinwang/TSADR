import torchvision.models as models
import torch
#from ptflops import get_model_complexity_info
from modules.Count import PCDNet
import numpy
import time


def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    N = sum([numpy.prod(p.size()) for p in parameters])
    return N

with torch.cuda.device(1):
    frame0 = torch.randn(1,3,512,512).cuda(1)
    frame2 = torch.randn(1,3,512,512).cuda(1)
    model = PCDNet().cuda(1)

    num = 50

    start = time.time() 
    for i in range(num):
    	out = model(frame0, frame2)
    end = time.time() 

    print("Num. of model parameters is :" + str(count_network_parameters(model)/(1000**2)))
    print('run time: %.3f s' % ((end -start)/num))
    #print('{:<30}  {:<8}'.format('Number of parameters: ', params))
