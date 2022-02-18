# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:17:29 2019

@author: JIAN
"""
import torch
import scipy.io as sio
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import mne
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec


torch.cuda.empty_cache()
torch.manual_seed(0)
np.random.seed(0)
plt.rcParams.update({'font.size': 14})

"""
The codes implement the paper "Towards Best Practice of Interpreting Deep Learning Models for EEG-based Brain Computer Interfaces ".
https://arxiv.org/abs/2202.06948  (if you find the codes useful pls cite our paper)

In this project, we implemented 7 interpretation techniques on two benchmark deep learning models "EEGNet" and "InterpretableCNN" for EEG-based BCI. 
The methods include:
    
gradient×input, 
DeepLIFT, 
integrated gradient, 
layer-wise relevance propagation (LRP),
saliency map, 
deconvolution,g
guided backpropagation 


The dataset for the test is available from 
https://figshare.com/articles/dataset/EEG_driver_drowsiness_dataset/14273687

The dataset contains 3 variables and they are EEGsample, substate and subindex.
 "EEGsample" contains 2022 EEG samples of size 20x384 from 11 subjects. 
 Each sample is a 3s EEG data with 128Hz from 30 EEG channels.

 The names and their corresponding index are shown below:
 Fp1, Fp2, F7, F3, Fz, F4, F8, FT7, FC3, FCZ, FC4, FT8, T3, C3, Cz, C4, T4, TP7, CP3, CPz, CP4, TP8, T5, P3, PZ, P4, T6, O1, Oz  O2
 0,    1,  2,  3,  4,  5,  6,  7,   8,   9,   10,   11, 12, 13, 14, 15, 16, 17,  18,  19,  20,  21,  22,  23,24, 25, 26, 27, 28, 29


 "subindex" is an array of 2022x1. It contains the subject indexes from 1-11 corresponding to each EEG sample. 
 "substate" is an array of 2022x1. It contains the labels of the samples. 0 corresponds to the alert state and 1 correspond to the drowsy state.
 



If you have met any problems, you can contact Dr. Cui Jian at cuij0006@ntu.edu.sg
    
"""




class InterpretableCNN(torch.nn.Module):  
    def __init__(self, classes=2, sampleChannel=30, sampleLength=384 ,N1=16, d=2,kernelLength=64):
        super(InterpretableCNN, self).__init__()
        
        # model layers
        self.pointwise = torch.nn.Conv2d(1,N1,(sampleChannel,1))
        self.depthwise = torch.nn.Conv2d(N1,d*N1,(1,kernelLength),groups=N1) 
        self.activ=torch.nn.ReLU()          
        self.batchnorm = torch.nn.BatchNorm2d(d*N1,track_running_stats=False)   
        self.GAP=torch.nn.AvgPool2d((1, sampleLength-kernelLength+1))         
        self.fc = torch.nn.Linear(d*N1, classes)        
        self.softmax=torch.nn.LogSoftmax(dim=1)


        # parameters for the interpretation techniques
        self.activ_in=0
        self.activ_out=0
        self.activ_baseline_in=0
        self.activ_baseline_out=0
        self.method=None
        self.DeepLIFT_baseline=False
        self.batch_mean=0
        self.batch_std=0
        self.gamma=0
        self.beta=0
        self.eps=1e-05 
        self.activalue=None


    def forward(self, inputdata):
        intermediate = self.pointwise(inputdata)         
        intermediate = self.depthwise(intermediate) 
        intermediate = self.activ(intermediate)  
 
        intermediate = self.batchnorm(intermediate)          
        intermediate = self.GAP(intermediate)     
        intermediate = intermediate.view(intermediate.size()[0], -1) 
        intermediate = self.fc(intermediate)    
        output = self.softmax(intermediate)   

        return output 
    

    #removal of the softmax layer for smooth back-propagation
    def update_softmax_forward(self):
        def softmax_forward_hook_function(module, ten_in, ten_out): 
            return ten_in[0]
        handle=self.softmax.register_forward_hook(softmax_forward_hook_function)  
        
        return handle

    # make the batch normalization layer a linear operation
    def update_batch_forward(self):
        def batch_forward_hook_function(module, ten_in, ten_out):   

            data=ten_in[0]
            batchmean=self.batch_mean.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3)))
            batchstd=self.batch_std.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3)))

            data=torch.div((ten_in[0]-batchmean.cuda()),batchstd.cuda())
            gammamatrix=(self.gamma.cuda()).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3)))
            betamatrix =(self.beta.cuda()).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3)))
            
            output=data*gammamatrix+betamatrix

            return output
                
        handle=self.batchnorm.register_forward_hook(batch_forward_hook_function)   
        
        return [handle]

    # savew the batch mean and std
    def update_batch_forward_meanstd(self):
        def batch_forward_hook_function(module, ten_in, ten_out):   

            data = ten_in[0].clone().detach().requires_grad_(False).cpu().double()

            self.batch_mean=torch.mean(data, [0,2,3],True)
            self.batch_std =torch.sqrt(torch.mean((data-self.batch_mean)**2, [0,2,3],True)+self.eps)
            
            self.gamma=torch.DoubleTensor(1,ten_in[0].size(1),1,1)
            self.beta=torch.DoubleTensor(1,ten_in[0].size(1),1,1)
            
            self.gamma[0,:,0,0]=self.batchnorm.weight.clone().detach().requires_grad_(False).cpu()
            self.beta[0,:,0,0]=self.batchnorm.bias.clone().detach().requires_grad_(False).cpu()
                        
        handle=self.batchnorm.register_forward_hook(batch_forward_hook_function)   
        return [handle]


    # save the values before and after the nonlinear layer in the forward pass
    def update_activ_forward(self):
        def activ_forward_hook_function(module, ten_in, ten_out):        
            self.activ_in=ten_in[0].clone().detach().requires_grad_(False).cpu()
            self.activ_out=ten_out.clone().detach().requires_grad_(False).cpu()      
   
        handle=self.activ.register_forward_hook(activ_forward_hook_function)   
        
        return [handle]
    
    
    # save the values before and after the nonlinear layer for the baseline input in the forward pass
    def update_activ_forward_baseline(self):
        def activ_forward_hook_function(module, ten_in, ten_out):        
            self.activ_baseline_in=ten_in[0].clone().detach().requires_grad_(False).cpu()
            self.activ_baseline_out=ten_out.clone().detach().requires_grad_(False).cpu()      
  
        handle=self.activ.register_forward_hook(activ_forward_hook_function)   
        return [handle]


    def update_activ_DeepLIFT(self):
        def activ_backward_hook_function(mmodule, grad_in, grad_out):
            delta_in=self.activ_baseline_in-self.activ_in
            delta_out=self.activ_baseline_out-self.activ_out
                
            modified_grad=torch.where(torch.abs(delta_in) > self.eps, grad_out[0].cpu() * torch.div(delta_out, delta_in),grad_in[0].cpu()).cuda()
            return (modified_grad,)

        handle=self.activ.register_backward_hook(activ_backward_hook_function) 
        return [handle]

    def update_activ_EpsilonLRP(self):
        def activ_backward_hook_function(mmodule, grad_in, grad_out):
            vabrate=self.eps *torch.where(self.activ_in >= 0, torch.ones_like(self.activ_in), -1 * torch.ones_like(self.activ_in)) 
            modified_grad=torch.div(grad_out[0].cpu()*self.activ_out,(self.activ_in + vabrate)).cuda()
                
            return (modified_grad,)
        
        handle=self.activ.register_backward_hook(activ_backward_hook_function)  
        return [handle]
    
        
    def update_activ_deconvolution(self):
        def activ_backward_hook_function(mmodule, grad_in, grad_out):

            modified_grad = torch.clamp(grad_out[0], min=0.0)          
            return (modified_grad,)
        
        handle=self.activ.register_backward_hook(activ_backward_hook_function)  
        return [handle]        
        
        
    def update_activ_guidedbackpropogation(self):
        def activ_backward_hook_function(mmodule, grad_in, grad_out):
            forwardpass = torch.where(self.activ_out>0, torch.ones_like(self.activ_out), torch.zeros_like(self.activ_out)).cuda()
            modified_grad =forwardpass * torch.clamp(grad_out[0], min=0.0)              
            return (modified_grad,)
        
        handle=self.activ.register_backward_hook(activ_backward_hook_function)  
        return [handle]  
    
   
    


class EEGNet(torch.nn.Module):
    def __init__(self,channelnum=30):
        super(EEGNet, self).__init__()

        # model parameters
        self.eps=1e-05 
        
        self.f1=8
        self.d=2
        self.conv1 = torch.nn.Conv2d(1,self.f1,(1,64),padding = (0,32),bias=False)
        self.batchnorm1 = torch.nn.BatchNorm2d(self.f1,track_running_stats=False)
        self.batchnorm2 = torch.nn.BatchNorm2d(self.f1*self.d,track_running_stats=False)
        self.batchnorm3 = torch.nn.BatchNorm2d(self.f1*self.d,track_running_stats=False) 
        self.activ1=torch.nn.ELU()
        self.activ2=torch.nn.ELU()
        self.depthconv = torch.nn.Conv2d(self.f1,self.f1*self.d,(30,1),groups=self.f1,bias=False)
        self.avgpool = torch.nn.AvgPool2d((1,4))
        self.separable = torch.nn.Conv2d(self.f1*self.d,self.f1*self.d,(1,16), padding = (0,8), groups=self.f1*self.d,bias=False)      
        self.fc1 = torch.nn.Linear(192, 2)   #128      
        self.softmax=nn.LogSoftmax(dim=1)
        self.softmax1=nn.Softmax(dim=1)
        self.dropout=nn.Dropout(p=0.5)
        
        # parameters for the interpretation techniques
        self.batch_mean1=0
        self.batch_std1=0
        self.gamma1=0
        self.beta1=0        
        self.batch_mean2=0
        self.batch_std2=0
        self.gamma2=0
        self.beta2=0         
        self.batch_mean3=0
        self.batch_std3=0
        self.gamma3=0
        self.beta3=0 
        self.activ_in1=0
        self.activ_out1=0
        self.activ_baseline_in1=0
        self.activ_baseline_out1=0
        self.activ_in2=0
        self.activ_out2=0
        self.activ_baseline_in2=0
        self.activ_baseline_out2=0

    def forward(self, inputdata):  
        intermediate = self.conv1(inputdata)           
        intermediate = self.batchnorm1(intermediate)
        intermediate = self.depthconv(intermediate)
        intermediate = self.batchnorm2(intermediate)
        intermediate = self.activ1(intermediate)
        intermediate = F.avg_pool2d(intermediate,(1, 4))
        intermediate =self.dropout(intermediate)
        intermediate = self.separable(intermediate)
        intermediate = self.batchnorm3(intermediate)
        intermediate = self.activ2(intermediate)
        intermediate = F.avg_pool2d(intermediate,(1, 8))   
        intermediate=self.dropout(intermediate)

        intermediate = intermediate.view(intermediate.size()[0], -1)
        intermediate = self.fc1(intermediate)
        output = self.softmax(intermediate)   

        return output  
  
    def update_softmax_forward(self):
        def softmax_forward_hook_function(module, ten_in, ten_out): 
            return ten_in[0]
        handle=self.softmax.register_forward_hook(softmax_forward_hook_function)  
        
        return handle    
    
    def update_batch_forward(self):
        def batch_forward_hook_function1(module, ten_in, ten_out):  
            data=ten_in[0]
            batchmean1=self.batch_mean1.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3)))
            batchstd1=self.batch_std1.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3)))
            
            data=torch.div((ten_in[0]-batchmean1.cuda()),batchstd1.cuda())
            gammamatrix=(self.gamma1.cuda()).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3)))
            betamatrix =(self.beta1.cuda()).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3)))
            
            output=data*gammamatrix+betamatrix

            return output
            
        def batch_forward_hook_function2(module, ten_in, ten_out): 
            data=ten_in[0]
            batchmean2=self.batch_mean2.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3)))
            batchstd2=self.batch_std2.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3))) 
        
            data=torch.div((ten_in[0]-batchmean2.cuda()),batchstd2.cuda())
            gammamatrix=(self.gamma2.cuda()).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3)))
            betamatrix =(self.beta2.cuda()).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3)))

            output=data*gammamatrix+betamatrix
          
            return output   
        
        def batch_forward_hook_function3(module, ten_in, ten_out):
        
            data=ten_in[0]
            batchmean3=self.batch_mean3.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3)))
            batchstd3=self.batch_std3.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3)))
                       
            data=torch.div((ten_in[0]-batchmean3.cuda()),batchstd3.cuda())
            gammamatrix=(self.gamma3.cuda()).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3)))
            betamatrix =(self.beta3.cuda()).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),int(data.size(3)))

            output=data*gammamatrix+betamatrix        
   
            return output      
        
   
        handle1=self.batchnorm1.register_forward_hook(batch_forward_hook_function1)   
        handle2=self.batchnorm2.register_forward_hook(batch_forward_hook_function2) 
        handle3=self.batchnorm3.register_forward_hook(batch_forward_hook_function3) 
        
        return [handle1,handle2,handle3]


    def update_batch_forward_meanstd(self):
        def batch_forward_hook_function1(module, ten_in, ten_out):   
            
            data = ten_in[0].clone().detach().requires_grad_(False).cpu().double()

            self.batch_mean1=torch.mean(data, [0,2,3],True)
            self.batch_std1 =torch.sqrt(torch.mean((data-self.batch_mean1)**2, [0,2,3],True)+self.eps)
                        
            
            self.gamma1=torch.DoubleTensor(1,ten_in[0].size(1),1,1)
            self.beta1=torch.DoubleTensor(1,ten_in[0].size(1),1,1)
            
            self.gamma1[0,:,0,0]=self.batchnorm1.weight.clone().detach().requires_grad_(False).cpu()
            self.beta1[0,:,0,0]=self.batchnorm1.bias.clone().detach().requires_grad_(False).cpu()

        def batch_forward_hook_function2(module, ten_in, ten_out):   

            data = ten_in[0].clone().detach().requires_grad_(False).cpu().double()

            self.batch_mean2=torch.mean(data, [0,2,3],True)
            self.batch_std2 =torch.sqrt(torch.mean((data-self.batch_mean2)**2, [0,2,3],True)+self.eps)
            
            self.gamma2=torch.DoubleTensor(1,ten_in[0].size(1),1,1)
            self.beta2=torch.DoubleTensor(1,ten_in[0].size(1),1,1)
            
            self.gamma2[0,:,0,0]=self.batchnorm2.weight.clone().detach().requires_grad_(False).cpu()
            self.beta2[0,:,0,0]=self.batchnorm2.bias.clone().detach().requires_grad_(False).cpu()
                        
        def batch_forward_hook_function3(module, ten_in, ten_out):   
            
            data = ten_in[0].clone().detach().requires_grad_(False).cpu().double()

            self.batch_mean3=torch.mean(data, [0,2,3],True)
            self.batch_std3 =torch.sqrt(torch.mean((data-self.batch_mean3)**2, [0,2,3],True)+self.eps)
            
            self.gamma3=torch.DoubleTensor(1,ten_in[0].size(1),1,1)
            self.beta3=torch.DoubleTensor(1,ten_in[0].size(1),1,1)
            
            self.gamma3[0,:,0,0]=self.batchnorm3.weight.clone().detach().requires_grad_(False).cpu()
            self.beta3[0,:,0,0]=self.batchnorm3.bias.clone().detach().requires_grad_(False).cpu()

   
        handle1=self.batchnorm1.register_forward_hook(batch_forward_hook_function1) 
        handle2=self.batchnorm2.register_forward_hook(batch_forward_hook_function2) 
        handle3=self.batchnorm3.register_forward_hook(batch_forward_hook_function3) 
        
        return [handle1,handle2,handle3]

    def update_activ_forward(self):
        def activ_forward_hook_function1(module, ten_in, ten_out):        
            self.activ_in1=ten_in[0].clone().detach().requires_grad_(False).cpu()
            self.activ_out1=ten_out.clone().detach().requires_grad_(False).cpu()  
            
        def activ_forward_hook_function2(module, ten_in, ten_out):        
            self.activ_in2=ten_in[0].clone().detach().requires_grad_(False).cpu()
            self.activ_out2=ten_out.clone().detach().requires_grad_(False).cpu() 

        handle1=self.activ1.register_forward_hook(activ_forward_hook_function1)   
        handle2=self.activ2.register_forward_hook(activ_forward_hook_function2) 
      #  
        return [handle1,handle2]
    
        
    def update_activ_forward_baseline(self):
        def activ_forward_hook_function1(module, ten_in, ten_out):        
            self.activ_baseline_in1=ten_in[0].clone().detach().requires_grad_(False).cpu()
            self.activ_baseline_out1=ten_out.clone().detach().requires_grad_(False).cpu()  
            
        def activ_forward_hook_function2(module, ten_in, ten_out):        
            self.activ_baseline_in2=ten_in[0].clone().detach().requires_grad_(False).cpu()
            self.activ_baseline_out2=ten_out.clone().detach().requires_grad_(False).cpu() 

        handle1=self.activ1.register_forward_hook(activ_forward_hook_function1)  
        handle2=self.activ2.register_forward_hook(activ_forward_hook_function2) 
         
        return [handle1,handle2]


    def update_activ_DeepLIFT(self):
        def activ_backward_hook_function1(mmodule, grad_in, grad_out):
            delta_in=self.activ_baseline_in1-self.activ_in1
            delta_out=self.activ_baseline_out1-self.activ_out1
                
            modified_grad=torch.where(torch.abs(delta_in) > self.eps, grad_out[0].cpu() * torch.div(delta_out, delta_in),grad_in[0].cpu()).cuda()
            return (modified_grad,)
        
        def activ_backward_hook_function2(mmodule, grad_in, grad_out):
            delta_in=self.activ_baseline_in2-self.activ_in2
            delta_out=self.activ_baseline_out2-self.activ_out2
                
            modified_grad=torch.where(torch.abs(delta_in) > self.eps, grad_out[0].cpu() * torch.div(delta_out, delta_in),grad_in[0].cpu()).cuda()
            return (modified_grad,)
        
        handle1=self.activ1.register_backward_hook(activ_backward_hook_function1)
        handle2=self.activ2.register_backward_hook(activ_backward_hook_function2)        
        return [handle1,handle2]

    def update_activ_EpsilonLRP(self):
        def activ_backward_hook_function1(mmodule, grad_in, grad_out):
            vabrate=self.eps *torch.where(self.activ_in1 >= 0, torch.ones_like(self.activ_in1), -1 * torch.ones_like(self.activ_in1)) 
            modified_grad=torch.div(grad_out[0].cpu()*self.activ_out1,(self.activ_in1 + vabrate)).cuda()
                
            return (modified_grad,)


        def activ_backward_hook_function2(mmodule, grad_in, grad_out):
            vabrate=self.eps *torch.where(self.activ_in2 >= 0, torch.ones_like(self.activ_in2), -1 * torch.ones_like(self.activ_in2)) 
            modified_grad=torch.div(grad_out[0].cpu()*self.activ_out2,(self.activ_in2 + vabrate)).cuda()
                
            return (modified_grad,)

        handle1=self.activ1.register_backward_hook(activ_backward_hook_function1) 
        handle2=self.activ2.register_backward_hook(activ_backward_hook_function2)  
        return [handle1,handle2]
    
        
    def update_activ_deconvolution(self):
        def activ_backward_hook_function(mmodule, grad_in, grad_out):
            modified_grad = torch.clamp(grad_out[0], min=0.0) 
            
            return (modified_grad,)
        
        handle1=self.activ1.register_backward_hook(activ_backward_hook_function)         
        handle2=self.activ2.register_backward_hook(activ_backward_hook_function)  
        return [handle1,handle2]       
        
        
    def update_activ_guidedbackpropogation(self):
        def activ_backward_hook_function1(mmodule, grad_in, grad_out):

            forwardpass = torch.where(self.activ_out1>0, torch.ones_like(self.activ_out1), torch.zeros_like(self.activ_out1)).cuda()
            modified_grad =forwardpass * torch.clamp(grad_out[0], min=0.0)  

            return (modified_grad,)


        def activ_backward_hook_function2(mmodule, grad_in, grad_out):

            forwardpass = torch.where(self.activ_out2>0, torch.ones_like(self.activ_out2), torch.zeros_like(self.activ_out2)).cuda()
            modified_grad =forwardpass * torch.clamp(grad_out[0], min=0.0)  

            return (modified_grad,)
        
        handle1=self.activ1.register_backward_hook(activ_backward_hook_function1)  
        handle2=self.activ2.register_backward_hook(activ_backward_hook_function2)  
        return [handle1,handle2] 

    
class VisTech():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        self.eps=0.000001        
        self.method=None      
    
    def enhanceheatmap(self, heatmap,r=5):
        
        sampleChannel=heatmap.shape[0]
        sampleLength=heatmap.shape[1]
        
        newmap=np.zeros((sampleChannel,sampleLength))
        for i in range(sampleChannel):
            for j in range(sampleLength):
                if j<r:
                    newmap[i,j]=np.mean(heatmap[i,:j+r])
                elif j+r>sampleLength:
                    newmap[i,j]=np.mean(heatmap[i,j-r:])        
                else:
                    newmap[i,j]=np.mean(heatmap[i,j-r:j+r])

        return newmap
    
    
    
    def convert_batchlayer_to_linear(self, batchInput):
            
        handles=self.model.update_batch_forward_meanstd()
        self.model(batchInput)
        self.remove_registered_functions(handles) 
        handles=self.model.update_batch_forward()
        
        return handles

        
    def remove_registered_functions(self, handles):
        for handle in handles:
            handle.remove()      
        
    def heatmap_calculation_backpropogation(self,batchInput,sampleidx,method='EpsilonLRP'):
        # This function output the heatmaps generate with different interpretation techniques. 
        # Most of the techques can be achieved by modifying the nonlinear activation layers
        
        def calculate_one_hot_out_put(output):
            result=output.cpu().detach().numpy()
            preds       = result.argmax(axis = -1)  
            one_hot_output = np.zeros(result.shape)
            
            for i in range(preds.shape[0]):
                one_hot_output[i,preds[i]]=1 
    
            one_hot_output =torch.DoubleTensor(one_hot_output).cuda()
            
            return one_hot_output

        sampleInput=batchInput
        sampleInput.requires_grad=True        
  
        handles0=self.convert_batchlayer_to_linear(batchInput)
        
        if method=="guidedbackpropogation":
            handles1=self.model.update_activ_forward()
            handles2=self.model.update_activ_guidedbackpropogation()            

            output=self.model(sampleInput)
            one_hot_output =calculate_one_hot_out_put(output)
            output.backward(gradient=one_hot_output)        
            grad=sampleInput.grad
            heatmap=grad.cpu().detach().numpy().squeeze() 
            
            self.remove_registered_functions(handles1+handles2)
            
        elif method=="random":
            heatmap=torch.rand(batchInput.size()).numpy().squeeze() 
            
        elif method=="EpsilonLRP":  
            handles1=self.model.update_activ_forward()
            handles2=self.model.update_activ_EpsilonLRP()               
            
            output=self.model(sampleInput)
            one_hot_output =calculate_one_hot_out_put(output)

            output.backward(gradient=one_hot_output)        
            grad=sampleInput.grad
            heatmap=torch.mul(grad,sampleInput)    
            heatmap=heatmap.cpu().detach().numpy().squeeze()
            self.remove_registered_functions(handles1+handles2)
            
        elif method=="GradxInput":

            output=self.model(sampleInput)
            one_hot_output =calculate_one_hot_out_put(output)
            
            output.backward(gradient=one_hot_output)        
            grad=sampleInput.grad
            heatmap=torch.mul(grad,sampleInput)    

            heatmap=heatmap.cpu().detach().numpy().squeeze()
                 
        elif method=="DeepLIFT":
                      
            baselineinput=torch.zeros_like(sampleInput).cuda()
            handles = self.model.update_activ_forward_baseline()
            output=self.model(baselineinput)
            
            self.remove_registered_functions(handles)
            
            handles1=self.model.update_activ_forward()
            handles2=self.model.update_activ_DeepLIFT()               
            output=self.model(sampleInput)
            one_hot_output =calculate_one_hot_out_put(output)
            
            output.backward(gradient=one_hot_output)        
            grad=sampleInput.grad
            heatmap=torch.mul(grad,sampleInput)    
            heatmap=heatmap.cpu().detach().numpy().squeeze()              

            self.remove_registered_functions(handles1+handles2)
            
        elif method=="deconvolution":     
            handles1=self.model.update_activ_deconvolution()
            output=self.model(sampleInput)

            one_hot_output =calculate_one_hot_out_put(output)
            output.backward(gradient=one_hot_output)        
            grad=sampleInput.grad

            heatmap=grad.cpu().detach().numpy().squeeze() 
            self.remove_registered_functions(handles1)


        elif method=="Saliencymap":    
            output=self.model(sampleInput)
            
            one_hot_output =calculate_one_hot_out_put(output)
            output.backward(gradient=one_hot_output)        
            grad=sampleInput.grad
            heatmap=grad.cpu().detach().numpy().squeeze()            
            
        elif method=="IntegratedGrad":  
            output=self.model(sampleInput)
            one_hot_output =calculate_one_hot_out_put(output)

            x1=batchInput.cpu().detach().numpy()
            x0=np.zeros_like(x1)
            sumheatmap=np.zeros_like(x1)

            steps=100
            for alpha in list(np.linspace(0.0, 1.0, num=steps)):
                x=x0+alpha*(x1-x0)
                inputbat=torch.from_numpy(x).cuda()
                inputbat.requires_grad=True

                self.model.zero_grad()  
                output=self.model(inputbat)
                output.backward(gradient=one_hot_output) 

                sumheatmap=sumheatmap+inputbat.grad.cpu().detach().numpy()
                
            heatmap=(sumheatmap/steps*x1).squeeze()    
            
        self.remove_registered_functions(handles0)
        
        # the methods will generate heatmaps for a batch, otherwise return the heatmap for a sample
        if sampleidx!=None:
            heatmap=heatmap[sampleidx]

        return heatmap        
        

    def perturbation_test(self,heatmap,heatmap_channel, batchInput,sampleidx,state):     
        
        sampleChannel=batchInput.size()[2]
        sampleLength=batchInput.size()[3] 

        handles0=self.convert_batchlayer_to_linear(batchInput)
        handle=self.model.update_softmax_forward()
        
######## perturbation of the sample heatmap ############################          
        inputsig=batchInput[sampleidx,0,:,:]  
        inputsig=torch.reshape(inputsig,(1,1,sampleChannel,sampleLength))   
        inputdata=inputsig.clone().detach().cpu().numpy()
  
        output=self.model(torch.DoubleTensor(inputdata).cuda())        
        prob=output[0,state].detach().cpu().numpy()
        
        # to perturbate the sample 100 times for each radius
        rand_num=100
        # the step for perturbation 
        step=0.1
         
        # save the correlation for patches from 0.1-0.5
        samplecor=[]
        
        # corresponding to 0.1-0.5 portion of data from a channel to be perturbed
        for l in range(1,6):
            # half radius of the perturbation patch
            radius_half=int(np.round(step*l*sampleLength/2))
            
            #stores the sum of contribution scores of the perturbed areas for all the samples
            heatlist=[]        
             
            # the channel to be purturbed
            chanidx=np.random.randint(sampleChannel, size=rand_num)
            
            # the time point to be purturbed
            timeidx=np.random.randint(sampleLength-2*radius_half, size=rand_num)
            
            # store all the purturbed samples 
            inputdata=np.zeros((rand_num,int(inputsig.size(1)), int(inputsig.size(2)),int(inputsig.size(3))))
        
            for k in range(rand_num):
                heatlist.append(np.sum(heatmap[chanidx[k],timeidx[k]:(timeidx[k]+2*radius_half)]))
                
                inputdata[k,0]=inputsig.clone().detach().cpu().numpy()
                inputdata[k,0,chanidx[k],timeidx[k]:(timeidx[k]+2*radius_half)]=0

            # get the results for the purturbed samples in a single batch
            output=self.model(torch.DoubleTensor(inputdata).cuda())
            output=output[:,state].detach().cpu().numpy()
            
            scorelist=prob-output
            heatlist=np.array(heatlist)
            
            # calculate the correlation
            corr=np.corrcoef(heatlist, scorelist)
            samplecor.append(corr[0,1])
        
        
######## perturbation of the channel heatmap ############################  
        # store all the purturbed samples     
        inputdata=np.zeros((sampleChannel,int(inputsig.size(1)), int(inputsig.size(2)),int(inputsig.size(3))))
        
        # each channel is perturbed once by setting its values to zeros
        for k in range(sampleChannel):
            inputdata[k,0]=inputsig.clone().detach().cpu().numpy()
            inputdata[k,0,k,:]=0

        output=self.model(torch.DoubleTensor(inputdata).cuda())
        output=output[:,state].detach().cpu().numpy()
        
        scorelist=prob-output
        
        corr=np.corrcoef(heatmap_channel, scorelist)
        channelcor=corr[0,1]    
        
        self.remove_registered_functions(handles0)  
        handle.remove()
        
        return samplecor, channelcor
        
        
    def deletion_test(self,heatmap,heatmap_channel, batchInput,sampleidx,state,samplethres=-1,channelthres=0):     
        
        sampleChannel=batchInput.size()[2]
        sampleLength=batchInput.size()[3] 
        handles0=self.convert_batchlayer_to_linear(batchInput)    
        
# deletion test on the sample heatmap        
        inputsig=batchInput[sampleidx,0,:,:]
        inputsig=torch.reshape(inputsig,(1,1,sampleChannel,sampleLength))   

        inputdata=inputsig.clone().detach().cpu().numpy()
        inputdata_channel=np.zeros((1,1,sampleChannel,sampleLength))
   
        for j in range(sampleChannel):
            for k in range(sampleLength):
                inputdata_channel[0,0,j,k]=inputdata[0,0,j,k]
            
        # record the total deleted number of deleted points    
        countnum=0
        # record the percent of data deleted from each channel
        samplechannel_percent=np.zeros(sampleChannel)
        
        # delete the points below the threshold
        for n in range(sampleChannel):
            countk=0
            for m in range(sampleLength):
                if heatmap[n,m]>samplethres:
                    countnum=countnum+1
                    countk=countk+1
                    inputdata[0,0,n,m]=0
                    
            samplechannel_percent[n]=countk/sampleLength

        output=self.model(torch.DoubleTensor(inputdata).cuda())
        probs=np.exp(output.detach().cpu().numpy())  
        
        sample_del=probs[:,state]
        sample_percent=countnum/(sampleChannel*sampleLength)
  
        
# deletion test on the channel heatmap            
        countchannel=0
        count_channel=[]
        
        # delete whole channels below the threshold
        for n in range(sampleChannel):
            if heatmap_channel[n]>channelthres:
                inputdata_channel[0,0,n,:]=0
                countchannel=countchannel+1
                count_channel.append(n)
                
        output=self.model(torch.DoubleTensor(inputdata_channel).cuda())
        probs=np.exp(output.detach().cpu().numpy())          
        
        channel_del=probs[:,state]
        
        self.remove_registered_functions(handles0)  
        count_channel=np.array(count_channel)
        
        return sample_del,sample_percent,samplechannel_percent,channel_del,count_channel
        


    def generate_interpretation(self, batchInput,sampleidx,subid,samplelabel,likelihood,method):
        """      
        input:
           batchInput:          all the samples in a batch for classification
           sampleidx:           the index of the sample
           subid:               the ID of the subject
           samplelabel:         the ground truth label of the sample
           likelihood:          the likelihood of the sample to be classified into alert and drowsy state 
           method:              the interpretation method to be used   
        """        
        
        if likelihood[0]>likelihood[1]:
            state=0
        else:
            state=1

        if samplelabel==0:
            labelstr='alert'
        else:
            labelstr='drowsy'        
        
        
        sampleInput=batchInput[sampleidx].cpu().detach().numpy().squeeze()
        sampleChannel=sampleInput.shape[0] 
        sampleLength=sampleInput.shape[1]        
        
        channelnames=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8','T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz','O2']
            
        heatmap_sample_thres=2        
        heatmap_channel_thres=1
      
        
# generate the original sample and channel contribution maps
        heatmap=self.heatmap_calculation_backpropogation(batchInput=batchInput,sampleidx=sampleidx,method=method)
        heatmap_channel=np.mean(heatmap, axis=1)

# get the results of the perturbation test
        samplecor, channelcor=self.perturbation_test(heatmap,heatmap_channel, batchInput,sampleidx,state)


# Step 1: normalization    
        heatmap= (heatmap-np.mean(heatmap)) / (np.std(heatmap))
        heatmap_channel = (heatmap_channel -np.mean(heatmap_channel )) / (np.std(heatmap_channel))
        
        
# Step 2: thresholding         
        heatmap_channel=heatmap_channel-heatmap_channel_thres
        heatmap=heatmap-heatmap_sample_thres
        
        # set values below lower bound of color map -1 to -1
        for u in range(sampleChannel):
            for l in range(sampleLength):
                if heatmap[u,l]<-1:
                    heatmap[u,l]=-1        
# Step 3: smoothing  
        smooth_factor=5
        heatmap=self.enhanceheatmap(heatmap,smooth_factor)
      
# get the results of the deletion test
        # set the thresholds for the portion of data to be deleted
        samplethres=-1
        channelthres=0
       
        sample_del,sample_percent,samplechannel_percent,channel_del,countchannel = self.deletion_test(heatmap,heatmap_channel, batchInput,sampleidx,samplelabel,samplethres,channelthres)
        
        # sort channel names by its portion of data deleted
        cn=np.argsort(-samplechannel_percent)
        per=np.sort(-samplechannel_percent)

        chnam=[]
        chnamstr=''
        for kk in countchannel:
            chnam.append(channelnames[kk])
            chnamstr=chnamstr+str(channelnames[kk])+' '     

# draw the figure
        rowdivide=4
        fig = plt.figure(figsize=(15,9))
        gridlayout = gridspec.GridSpec(ncols=2, nrows=rowdivide, figure=fig,wspace=0.05, hspace=0.3)
        axs0 = fig.add_subplot(gridlayout[0:rowdivide-1,0])
        axs1 = fig.add_subplot(gridlayout[0:rowdivide-1,1])  
        axs2=fig.add_subplot(gridlayout[rowdivide-1,:])  
     
        axs2.xaxis.set_ticks([])
        axs2.yaxis.set_ticks([])
        
# display the evaluation results        
        axs2.text(0.01, 0.8, 'Model: InterpretableCNN     Interpretation: '+method+'     Smooth factor: '+str(smooth_factor)+'     Thresholds:['+str(heatmap_sample_thres)+', '+str(channelthres)+']',horizontalalignment='left',fontsize=15 )
        axs2.text(0.01, 0.6, 'Pertubation test(0.1-0.5): '+str(np.round(samplecor[0],2))+' '+str(np.round(samplecor[1],2))+' '+str(np.round(samplecor[2],2))+' '+str(np.round(samplecor[3],2))+' '+str(np.round(samplecor[4],2)),horizontalalignment='left',fontsize=15 )
        axs2.text(0.5, 0.6, 'Pertubation test(channel): '+str(np.round(channelcor,2)),fontsize=15)        
        axs2.text(0.01, 0.4, 'Deletion test(sample): '+str(np.round(sample_del[0],2))+'        '+'Total data deleted: '+str(np.round(sample_percent,2))+'           '+'Top 3 channels: '+channelnames[cn[0]]+'(' +str(np.round(-per[0],2))+') '+channelnames[cn[1]]+'(' +str(np.round(-per[1],2))+') '+channelnames[cn[2]]+'(' +str(np.round(-per[2],2))+')',fontsize=15 ) 
        axs2.text(0.01, 0.2, 'Deletion test(channel): '+str(np.round(channel_del[0],2))+'           '+'Deleted channels:'+chnamstr,fontsize=15 )    
    
        fig.suptitle('Subject:'+str(int(subid))+'   '+'Label:'+labelstr+'   '+'$P_{alert}=$'+str(round(likelihood[0],2))+'   $P_{drowsy}=$'+str(round(likelihood[1],2)),y=0.985,fontsize=17)  
        thespan=np.percentile(sampleInput,98)        
        xx=np.arange(1,sampleLength+1)  
                 
        for i in range(0,sampleChannel):            
            y=sampleInput[i,:]+thespan*(sampleChannel-1-i)
            dydx=heatmap[i,:]           
          
            points = np.array([xx, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(-1, 1)
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(dydx)
            lc.set_linewidth(2)
            axs0.add_collection(lc)
        
        
        yttics=np.zeros(sampleChannel)
        for gi in range(sampleChannel):
            yttics[gi]=gi*thespan

        axs0.set_ylim([-thespan,thespan*sampleChannel])          
        axs0.set_xlim([0,sampleLength+1]) 
        axs0.set_xticks([1,128,256,384])        
        axs0.set_xticklabels(['0','1','2','3(s)'])  

        inversechannelnames=[]
        for i in range(sampleChannel):
            inversechannelnames.append(channelnames[sampleChannel-1-i])
                   
        plt.sca(axs0)
        plt.yticks(yttics, inversechannelnames)        
        
        montage ='standard_1020'
        sfreq = 128
        
        info = mne.create_info(
            channelnames,
            ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg'],
            sfreq=sfreq,          
            montage=montage
        )

        im,cn=mne.viz.plot_topomap(data=heatmap_channel,pos=info, vmin=-1, vmax=1, axes=axs1, names=channelnames,show_names=True,outlines='head',cmap='viridis',show=False)
        fig.colorbar(im,ax=axs1)

  
    
    
def run():
    filename = r'dataset.mat'

    tmp = sio.loadmat(filename)
    xdata=np.array(tmp['EEGsample'])
    label=np.array(tmp['substate'])
    subIdx=np.array(tmp['subindex'])

    label.astype(int)
    subIdx.astype(int)
    
    samplenum=label.shape[0]

#   there are 11 subjects in the dataset. Each sample is 3-seconds data from 30 channels with sampling rate of 128Hz. 
    channelnum=30
    subjnum=11
    samplelength=3
    sf=128
    
#   define the learning rate, batch size and epoches
    lr=1e-3 
    batch_size = 50
    n_epoch =11

#   ydata contains the label of samples   
    ydata=np.zeros(samplenum,dtype=np.longlong)
    for i in range(samplenum):
        ydata[i]=label[i] 
        
# select the subject index here        
    for i in range(1,2):  
#       form the training data        
        trainindx=np.where(subIdx != i)[0] 
        xtrain=xdata[trainindx]   
        x_train = xtrain.reshape(xtrain.shape[0],1,channelnum, samplelength*sf)
        y_train=ydata[trainindx]
                        
#       form the testing data  
        testindx=np.where(subIdx == i)[0]    
        xtest=xdata[testindx]
        x_test = xtest.reshape(xtest.shape[0], 1,channelnum, samplelength*sf)
        y_test=ydata[testindx]

        train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

# select the deep learning model to be used
        my_net = InterpretableCNN().double().cuda()
   #     my_net = EEGNet().double().cuda()
        
        for p in my_net.parameters():
            p.requires_grad = True 
              
        optimizer = optim.Adam(my_net.parameters(), lr=lr)    
        loss_class = torch.nn.NLLLoss().cuda()

# train the classifier 
        for epoch in range(n_epoch):   
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data                
                
                input_data = inputs.cuda()
                class_label = labels.cuda()              

                my_net.zero_grad()               
                my_net.train()          
   
                class_output= my_net(input_data) 
                err_s_label = loss_class(class_output, class_label)
                err = err_s_label 
             
                err.backward()
                optimizer.step()

        
        my_net.eval()
        with torch.no_grad():
            x_test =  torch.DoubleTensor(x_test).cuda()
            answer = my_net(x_test)
            probs=np.exp(answer.cpu().numpy())
            
        sampleVis =VisTech(my_net)


# select the interpretation method to be used
#        method="random"         
#        method="Saliencymap"
#        method="GradxInput"        
#        method="deconvolution"
#        method="guidedbackpropogation"         
#        method="DeepLIFT"
#        method="IntegratedGrad"        
        method="EpsilonLRP" 
########################################
        
        sampleidx=98     
        sampleVis.generate_interpretation(batchInput=x_test,sampleidx=sampleidx,subid=i,samplelabel=y_test[sampleidx],likelihood=probs[sampleidx],method=method)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    run()
    
