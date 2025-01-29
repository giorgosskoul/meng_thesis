# -*- coding: utf-8 -*-
"""
CNN Models for 2D Multiclass-Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor


## CNN model with 1 Convolutional Layer and 1 Fully-Connected Layer
class cnn11(nn.Module):
    def __init__(self,in_channels=1,no_classes=8, img_dim=224,
                 conv_output=16,conv_kernel=2,conv_stride=1,conv_padding=1,
                 pool_kernel=2,pool_stride=2,
                 print_option=0):
        ### if print_option is set to 1 then the dimensions of 
        ### every input & output will be printed.
        super(cnn11, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=conv_output, 
                               kernel_size=conv_kernel, 
                               stride=conv_stride, 
                               padding=conv_padding)
        out1=floor((img_dim+2*conv_padding-conv_kernel)/conv_stride)+1
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, 
                                 stride=pool_stride)
        out2=floor((out1-pool_kernel)/pool_stride)+1
        # Define the fully connected layers
        self.fc1 = nn.Linear(conv_output*out2*out2, no_classes)
        self.print_option=print_option
        self.out_channels=conv_output
        self.out_dim=out2
        
    def forward(self, x):
        x0=x
        # Apply the convolutional layers with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x1=x
        x = self.pool(x)
        x2=x
        # Reshape the tensor to feed into the fully connected layers
        x = x.view(-1, self.out_channels*self.out_dim*self.out_dim)
        x3=x
        x = self.fc1(x)
        # Softmax will be calculated with the CrossEntropy function...
        ## if print_option is 1:
        if self.print_option==1:
            print("--------DIMENSIONS-----------")
            print("Input : ",list(x0.shape))
            print("After Conv + Relu : ",list(x1.shape))
            print("After Pool : ",list(x2.shape))
            print("After Flattening : ",list(x3.shape))
            print("Output - After Fully-connected : ",list(x.shape))
            self.print_option=0
        return x
 
  
#---------------------------------------------------------------------#

  
## CNN model with 3 Convolutional Layers and 1 Fully-Connected Layer
class cnn31(nn.Module):
    def __init__(self,in_channels=1,no_classes=8, img_dim=224,
                 conv1_output=16,conv1_kernel=2,conv1_stride=1,conv1_padding=1,
                 conv2_output=32,conv2_kernel=2,conv2_stride=1,conv2_padding=1,
                 conv3_output=64,conv3_kernel=2,conv3_stride=1,conv3_padding=1,
                 pool1_kernel=2,pool1_stride=2,
                 pool2_kernel=2,pool2_stride=2,
                 pool3_kernel=2,pool3_stride=2,
                 print_option=0):
        ### if print_option is set to 1 then the dimensions of 
        ### every input & output will be printed.
        super(cnn31, self).__init__()
        # Define the convolutional layers
        ## CONV1
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=conv1_output, 
                               kernel_size=conv1_kernel, 
                               stride=conv1_stride, 
                               padding=conv1_padding)
        out1=floor((img_dim+2*conv1_padding-conv1_kernel)/conv1_stride)+1
        # Define the max pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=pool1_kernel, 
                                 stride=pool1_stride)
        out2=floor((out1-pool1_kernel)/pool1_stride)+1
        ## CONV2
        self.conv2= nn.Conv2d(in_channels=conv1_output, 
                               out_channels=conv2_output, 
                               kernel_size=conv2_kernel, 
                               stride=conv2_stride, 
                               padding=conv2_padding)
        out3=floor((out2+2*conv2_padding-conv2_kernel)/conv2_stride)+1
        # Define the max pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=pool2_kernel, 
                                 stride=pool2_stride)
        out4=floor((out3-pool2_kernel)/pool2_stride)+1
        ## CONV3
        self.conv3 = nn.Conv2d(in_channels=conv2_output, 
                               out_channels=conv3_output, 
                               kernel_size=conv3_kernel, 
                               stride=conv3_stride, 
                               padding=conv3_padding)
        out5=floor((out4+2*conv3_padding-conv3_kernel)/conv3_stride)+1
        # Define the max pooling layer
        self.pool3 = nn.MaxPool2d(kernel_size=pool3_kernel, 
                                 stride=pool3_stride)
        out6=floor((out5-pool3_kernel)/pool3_stride)+1
        # Define the fully connected layers
        self.fc1 = nn.Linear(conv3_output*out6*out6, no_classes)
        self.print_option=print_option
        self.out_channels=conv3_output
        self.out_dim=out6
        
    def forward(self, x):
        x0=x
        # Apply the convolutional layers with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x1=x
        x = self.pool1(x)
        x2=x
        x = F.relu(self.conv2(x))
        x3=x
        x = self.pool2(x)
        x4=x
        x = F.relu(self.conv3(x))
        x5=x
        x = self.pool3(x)
        x6=x
        # Reshape the tensor to feed into the fully connected layers
        x = x.view(-1, self.out_channels*self.out_dim*self.out_dim)
        x7=x
        x = self.fc1(x)
        # Softmax will be calculated with the CrossEntropy function...
        ## if print_option is 1:
        if self.print_option==1:
            print("--------DIMENSIONS-----------")
            print("Input : ",list(x0.shape))
            print("After Conv1 + Relu : ",list(x1.shape))
            print("After Pool1 : ",list(x2.shape))
            print("After Conv2 + Relu : ",list(x3.shape))
            print("After Pool2 : ",list(x4.shape))
            print("After Conv3 + Relu : ",list(x5.shape))
            print("After Pool3 : ",list(x6.shape))
            print("After Flattening : ",list(x7.shape))
            print("Output - After Fully-connected : ",list(x.shape))
            self.print_option=0
        return x
    
  
#---------------------------------------------------------------------#

## CNN model with 3 Convolutional Layers and 2 Fully-Connected Layers
class cnn32(nn.Module):
    def __init__(self,in_channels=1,no_classes=8, img_dim=224,
                 conv1_output=32,conv1_kernel=2,conv1_stride=1,conv1_padding=1,
                 conv2_output=32,conv2_kernel=2,conv2_stride=1,conv2_padding=1,
                 conv3_output=64,conv3_kernel=2,conv3_stride=1,conv3_padding=1,
                 pool1_kernel=2,pool1_stride=2,
                 pool2_kernel=2,pool2_stride=2,
                 pool3_kernel=2,pool3_stride=2,
                 fc1_output=50,
                 print_option=0):
        ### if print_option is set to 1 then the dimensions of 
        ### every input & output will be printed.
        super(cnn32, self).__init__()
        # Define the convolutional layers
        ## CONV1
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=conv1_output, 
                               kernel_size=conv1_kernel, 
                               stride=conv1_stride, 
                               padding=conv1_padding)
        out1=floor((img_dim+2*conv1_padding-conv1_kernel)/conv1_stride)+1
        # Define the max pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=pool1_kernel, 
                                 stride=pool1_stride)
        out2=floor((out1-pool1_kernel)/pool1_stride)+1
        ## CONV2
        self.conv2= nn.Conv2d(in_channels=conv1_output, 
                               out_channels=conv2_output, 
                               kernel_size=conv2_kernel, 
                               stride=conv2_stride, 
                               padding=conv2_padding)
        out3=floor((out2+2*conv2_padding-conv2_kernel)/conv2_stride)+1
        # Define the max pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=pool2_kernel, 
                                 stride=pool2_stride)
        out4=floor((out3-pool2_kernel)/pool2_stride)+1
        ## CONV3
        self.conv3 = nn.Conv2d(in_channels=conv2_output, 
                               out_channels=conv3_output, 
                               kernel_size=conv3_kernel, 
                               stride=conv3_stride, 
                               padding=conv3_padding)
        out5=floor((out4+2*conv3_padding-conv3_kernel)/conv3_stride)+1
        # Define the max pooling layer
        self.pool3 = nn.MaxPool2d(kernel_size=pool3_kernel, 
                                 stride=pool3_stride)
        out6=floor((out5-pool3_kernel)/pool3_stride)+1
        # Define the fully connected layers
        self.fc1 = nn.Linear(conv3_output*out6*out6, fc1_output)
        self.fc2 = nn.Linear(fc1_output, no_classes)
        self.print_option=print_option
        self.out_channels=conv3_output
        self.out_dim=out6
        
    def forward(self, x):
        x0=x
        # Apply the convolutional layers with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x1=x
        x = self.pool1(x)
        x2=x
        x = F.relu(self.conv2(x))
        x3=x
        x = self.pool2(x)
        x4=x
        x = F.relu(self.conv3(x))
        x5=x
        x = self.pool3(x)
        x6=x
        # Reshape the tensor to feed into the fully connected layers
        x = x.view(-1, self.out_channels*self.out_dim*self.out_dim)
        x7=x
        x = F.relu(self.fc1(x))
        x8=x
        x = self.fc2(x)
        # Softmax will be calculated with the CrossEntropy function...
        ## if print_option is 1:
        if self.print_option==1:
            print("--------DIMENSIONS-----------")
            print("Input : ",list(x0.shape))
            print("After Conv1 + Relu : ",list(x1.shape))
            print("After Pool1 : ",list(x2.shape))
            print("After Conv2 + Relu : ",list(x3.shape))
            print("After Pool2 : ",list(x4.shape))
            print("After Conv3 + Relu : ",list(x5.shape))
            print("After Pool3 : ",list(x6.shape))
            print("After Flattening : ",list(x7.shape))
            print("After Fully-connected1 : ",list(x8.shape))
            print("Output - After Fully-connected2 : ",list(x.shape))
            self.print_option=0
        return x
 

#---------------------------------------------------------------------#





## CNN model with 3 Convolutional Layers and 3 Fully-Connected Layers
class cnn33(nn.Module):
    def __init__(self,in_channels=1,no_classes=8, img_dim=224,
                 conv1_output=32,conv1_kernel=2,conv1_stride=1,conv1_padding=1,
                 conv2_output=32,conv2_kernel=2,conv2_stride=1,conv2_padding=1,
                 conv3_output=64,conv3_kernel=2,conv3_stride=1,conv3_padding=1,
                 pool1_kernel=2,pool1_stride=2,
                 pool2_kernel=2,pool2_stride=2,
                 pool3_kernel=2,pool3_stride=2,
                 fc1_output=50,
                 fc2_output=50,
                 print_option=0):
        ### if print_option is set to 1 then the dimensions of 
        ### every input & output will be printed.
        super(cnn33, self).__init__()
        # Define the convolutional layers
        ## CONV1
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=conv1_output, 
                               kernel_size=conv1_kernel, 
                               stride=conv1_stride, 
                               padding=conv1_padding)
        out1=floor((img_dim+2*conv1_padding-conv1_kernel)/conv1_stride)+1
        # Define the max pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=pool1_kernel, 
                                 stride=pool1_stride)
        out2=floor((out1-pool1_kernel)/pool1_stride)+1
        ## CONV2
        self.conv2= nn.Conv2d(in_channels=conv1_output, 
                               out_channels=conv2_output, 
                               kernel_size=conv2_kernel, 
                               stride=conv2_stride, 
                               padding=conv2_padding)
        out3=floor((out2+2*conv2_padding-conv2_kernel)/conv2_stride)+1
        # Define the max pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=pool2_kernel, 
                                 stride=pool2_stride)
        out4=floor((out3-pool2_kernel)/pool2_stride)+1
        ## CONV3
        self.conv3 = nn.Conv2d(in_channels=conv2_output, 
                               out_channels=conv3_output, 
                               kernel_size=conv3_kernel, 
                               stride=conv3_stride, 
                               padding=conv3_padding)
        out5=floor((out4+2*conv3_padding-conv3_kernel)/conv3_stride)+1
        # Define the max pooling layer
        self.pool3 = nn.MaxPool2d(kernel_size=pool3_kernel, 
                                 stride=pool3_stride)
        out6=floor((out5-pool3_kernel)/pool3_stride)+1
        # Define the fully connected layers
        self.fc1 = nn.Linear(conv3_output*out6*out6, fc1_output)
        self.fc2 = nn.Linear(fc1_output, fc2_output)
        self.fc3 = nn.Linear(fc2_output, no_classes)
        self.print_option=print_option
        self.out_channels=conv3_output
        self.out_dim=out6
        
    def forward(self, x):
        x0=x
        # Apply the convolutional layers with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x1=x
        x = self.pool1(x)
        x2=x
        x = F.relu(self.conv2(x))
        x3=x
        x = self.pool2(x)
        x4=x
        x = F.relu(self.conv3(x))
        x5=x
        x = self.pool3(x)
        x6=x
        # Reshape the tensor to feed into the fully connected layers
        x = x.view(-1, self.out_channels*self.out_dim*self.out_dim)
        x7=x
        x = F.relu(self.fc1(x))
        x8=x
        x = F.relu(self.fc2(x))
        x9=x
        x = self.fc3(x)
        # Softmax will be calculated with the CrossEntropy function...
        ## if print_option is 1:
        if self.print_option==1:
            print("--------DIMENSIONS-----------")
            print("Input : ",list(x0.shape))
            print("After Conv1 + Relu : ",list(x1.shape))
            print("After Pool1 : ",list(x2.shape))
            print("After Conv2 + Relu : ",list(x3.shape))
            print("After Pool2 : ",list(x4.shape))
            print("After Conv3 + Relu : ",list(x5.shape))
            print("After Pool3 : ",list(x6.shape))
            print("After Flattening : ",list(x7.shape))
            print("After Fully-connected1 : ",list(x8.shape))
            print("After Fully-connected2 : ",list(x9.shape))
            print("Output - After Fully-connected3 : ",list(x.shape))
            self.print_option=0
        return x
 

#---------------------------------------------------------------------#




## CNN model with 4 Convolutional Layers and 2 Fully-Connected Layers
class cnn42(nn.Module):
    def __init__(self,in_channels=1,no_classes=8, img_dim=224,
                 conv1_output=32,conv1_kernel=2,conv1_stride=1,conv1_padding=1,
                 conv2_output=32,conv2_kernel=2,conv2_stride=1,conv2_padding=1,
                 conv3_output=64,conv3_kernel=2,conv3_stride=1,conv3_padding=1,
                 conv4_output=64,conv4_kernel=2,conv4_stride=1,conv4_padding=1,
                 pool1_kernel=2,pool1_stride=2,
                 pool2_kernel=2,pool2_stride=2,
                 pool3_kernel=2,pool3_stride=2,
                 pool4_kernel=2,pool4_stride=2,
                 fc1_output=50,
                 print_option=0):
        ### if print_option is set to 1 then the dimensions of 
        ### every input & output will be printed.
        super(cnn42, self).__init__()
        # Define the convolutional layers
        ## CONV1
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=conv1_output, 
                               kernel_size=conv1_kernel, 
                               stride=conv1_stride, 
                               padding=conv1_padding)
        out1=floor((img_dim+2*conv1_padding-conv1_kernel)/conv1_stride)+1
        # Define the max pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=pool1_kernel, 
                                 stride=pool1_stride)
        out2=floor((out1-pool1_kernel)/pool1_stride)+1
        ## CONV2
        self.conv2= nn.Conv2d(in_channels=conv1_output, 
                               out_channels=conv2_output, 
                               kernel_size=conv2_kernel, 
                               stride=conv2_stride, 
                               padding=conv2_padding)
        out3=floor((out2+2*conv2_padding-conv2_kernel)/conv2_stride)+1
        # Define the max pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=pool2_kernel, 
                                 stride=pool2_stride)
        out4=floor((out3-pool2_kernel)/pool2_stride)+1
        ## CONV3
        self.conv3 = nn.Conv2d(in_channels=conv2_output, 
                               out_channels=conv3_output, 
                               kernel_size=conv3_kernel, 
                               stride=conv3_stride, 
                               padding=conv3_padding)
        out5=floor((out4+2*conv3_padding-conv3_kernel)/conv3_stride)+1
        # Define the max pooling layer
        self.pool3 = nn.MaxPool2d(kernel_size=pool3_kernel, 
                                 stride=pool3_stride)
        out6=floor((out5-pool3_kernel)/pool3_stride)+1
        ## CONV4
        self.conv4 = nn.Conv2d(in_channels=conv3_output, 
                               out_channels=conv4_output, 
                               kernel_size=conv4_kernel, 
                               stride=conv4_stride, 
                               padding=conv4_padding)
        out7=floor((out6+2*conv4_padding-conv4_kernel)/conv4_stride)+1
        # Define the max pooling layer
        self.pool4 = nn.MaxPool2d(kernel_size=pool4_kernel, 
                                 stride=pool4_stride)
        out8=floor((out7-pool4_kernel)/pool4_stride)+1
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(conv4_output*out8*out8, fc1_output)
        self.fc2 = nn.Linear(fc1_output, no_classes)
        self.print_option=print_option
        self.out_channels=conv4_output
        self.out_dim=out8
        
    def forward(self, x):
        x0=x
        # Apply the convolutional layers with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x1=x
        x = self.pool1(x)
        x2=x
        x = F.relu(self.conv2(x))
        x3=x
        x = self.pool2(x)
        x4=x
        x = F.relu(self.conv3(x))
        x5=x
        x = self.pool3(x)
        x6=x
        x = F.relu(self.conv4(x))
        x7=x
        x = self.pool4(x)
        x8=x
        # Reshape the tensor to feed into the fully connected layers
        x = x.view(-1, self.out_channels*self.out_dim*self.out_dim)
        x9=x
        x = F.relu(self.fc1(x))
        x10=x
        x = self.fc2(x)
        # Softmax will be calculated with the CrossEntropy function...
        ## if print_option is 1:
        if self.print_option==1:
            print("--------DIMENSIONS-----------")
            print("Input : ",list(x0.shape))
            print("After Conv1 + Relu : ",list(x1.shape))
            print("After Pool1 : ",list(x2.shape))
            print("After Conv2 + Relu : ",list(x3.shape))
            print("After Pool2 : ",list(x4.shape))
            print("After Conv3 + Relu : ",list(x5.shape))
            print("After Pool3 : ",list(x6.shape))
            print("After Conv4 + Relu : ",list(x7.shape))
            print("After Pool4 : ",list(x8.shape))
            print("After Flattening : ",list(x9.shape))
            print("After Fully-connected1 : ",list(x10.shape))
            print("Output - After Fully-connected2 : ",list(x.shape))
            self.print_option=0
        return x
 


#---------------------------------------------------------------------#



## CNN model with 4 Convolutional Layers and 3 Fully-Connected Layers
class cnn43(nn.Module):
    def __init__(self,in_channels=1,no_classes=8, img_dim=224,
                 conv1_output=32,conv1_kernel=2,conv1_stride=1,conv1_padding=1,
                 conv2_output=32,conv2_kernel=2,conv2_stride=1,conv2_padding=1,
                 conv3_output=64,conv3_kernel=2,conv3_stride=1,conv3_padding=1,
                 conv4_output=64,conv4_kernel=2,conv4_stride=1,conv4_padding=1,
                 pool1_kernel=2,pool1_stride=2,
                 pool2_kernel=2,pool2_stride=2,
                 pool3_kernel=2,pool3_stride=2,
                 pool4_kernel=2,pool4_stride=2,
                 fc1_output=50,fc2_output=50,
                 print_option=0):
        ### if print_option is set to 1 then the dimensions of 
        ### every input & output will be printed.
        super(cnn43, self).__init__()
        # Define the convolutional layers
        ## CONV1
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=conv1_output, 
                               kernel_size=conv1_kernel, 
                               stride=conv1_stride, 
                               padding=conv1_padding)
        out1=floor((img_dim+2*conv1_padding-conv1_kernel)/conv1_stride)+1
        # Define the max pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=pool1_kernel, 
                                 stride=pool1_stride)
        out2=floor((out1-pool1_kernel)/pool1_stride)+1
        ## CONV2
        self.conv2= nn.Conv2d(in_channels=conv1_output, 
                               out_channels=conv2_output, 
                               kernel_size=conv2_kernel, 
                               stride=conv2_stride, 
                               padding=conv2_padding)
        out3=floor((out2+2*conv2_padding-conv2_kernel)/conv2_stride)+1
        # Define the max pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=pool2_kernel, 
                                 stride=pool2_stride)
        out4=floor((out3-pool2_kernel)/pool2_stride)+1
        ## CONV3
        self.conv3 = nn.Conv2d(in_channels=conv2_output, 
                               out_channels=conv3_output, 
                               kernel_size=conv3_kernel, 
                               stride=conv3_stride, 
                               padding=conv3_padding)
        out5=floor((out4+2*conv3_padding-conv3_kernel)/conv3_stride)+1
        # Define the max pooling layer
        self.pool3 = nn.MaxPool2d(kernel_size=pool3_kernel, 
                                 stride=pool3_stride)
        out6=floor((out5-pool3_kernel)/pool3_stride)+1
        ## CONV4
        self.conv4 = nn.Conv2d(in_channels=conv3_output, 
                               out_channels=conv4_output, 
                               kernel_size=conv4_kernel, 
                               stride=conv4_stride, 
                               padding=conv4_padding)
        out7=floor((out6+2*conv4_padding-conv4_kernel)/conv4_stride)+1
        # Define the max pooling layer
        self.pool4 = nn.MaxPool2d(kernel_size=pool4_kernel, 
                                 stride=pool4_stride)
        out8=floor((out7-pool4_kernel)/pool4_stride)+1
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(conv4_output*out8*out8, fc1_output)
        self.fc2 = nn.Linear(fc1_output,fc2_output)
        self.fc3 = nn.Linear(fc2_output, no_classes)
        self.print_option=print_option
        self.out_channels=conv4_output
        self.out_dim=out8
        
    def forward(self, x):
        x0=x
        # Apply the convolutional layers with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x1=x
        x = self.pool1(x)
        x2=x
        x = F.relu(self.conv2(x))
        x3=x
        x = self.pool2(x)
        x4=x
        x = F.relu(self.conv3(x))
        x5=x
        x = self.pool3(x)
        x6=x
        x = F.relu(self.conv4(x))
        x7=x
        x = self.pool4(x)
        x8=x
        # Reshape the tensor to feed into the fully connected layers
        x = x.view(-1, self.out_channels*self.out_dim*self.out_dim)
        x9=x
        x = F.relu(self.fc1(x))
        x10=x
        x = F.relu(self.fc2(x))
        x11=x
        x = self.fc3(x)
        # Softmax will be calculated with the CrossEntropy function...
        ## if print_option is 1:
        if self.print_option==1:
            print("--------DIMENSIONS-----------")
            print("Input : ",list(x0.shape))
            print("After Conv1 + Relu : ",list(x1.shape))
            print("After Pool1 : ",list(x2.shape))
            print("After Conv2 + Relu : ",list(x3.shape))
            print("After Pool2 : ",list(x4.shape))
            print("After Conv3 + Relu : ",list(x5.shape))
            print("After Pool3 : ",list(x6.shape))
            print("After Conv4 + Relu : ",list(x7.shape))
            print("After Pool4 : ",list(x8.shape))
            print("After Flattening : ",list(x9.shape))
            print("After Fully-connected1 : ",list(x10.shape))
            print("After Fully-connected2 : ",list(x11.shape))
            print("Output - After Fully-connected3 : ",list(x.shape))
            self.print_option=0
        return x
 


# POKENETS #


class BulbaNet(nn.Module):
    def __init__(self, num_classes=8, num_channels=1, img_dim=40,
                 conv1_output=128,conv1_kernel=2,conv1_stride=1,conv1_padding=1,
                 conv2_output=256,conv2_kernel=2,conv2_stride=1,conv2_padding=1,
                 conv3_output=390,conv3_kernel=2,conv3_stride=1,conv3_padding=1,     
                 pool1_kernel=2,pool1_stride=2,
                 pool2_kernel=2,pool2_stride=2,
                 pool3_kernel=2,pool3_stride=2,
                 fc1_output=2048,fc2_output=2048,
                 bn_momentum=0.1,
                 print_option=0):
        super(BulbaNet, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=num_channels, 
                               out_channels=conv1_output, 
                               kernel_size=conv1_kernel, stride=conv1_stride, 
                               padding=conv1_padding)
        
        self.conv2 = nn.Conv2d(in_channels=conv1_output, 
                               out_channels=conv2_output, 
                               kernel_size=conv2_kernel, stride=conv2_stride, 
                               padding=conv2_padding)
        self.conv3 = nn.Conv2d(in_channels=conv2_output, 
                               out_channels=conv3_output, 
                               kernel_size=conv3_kernel, stride=conv3_stride, 
                               padding=conv3_padding)
        
        # Define the average pooling layers
        self.avgpool1 = nn.AvgPool2d(kernel_size=pool1_kernel, stride=pool1_stride)
        self.avgpool2 = nn.AvgPool2d(kernel_size=pool2_kernel, stride=pool2_stride)
        self.avgpool3 = nn.AvgPool2d(kernel_size=pool3_kernel, stride=pool3_stride)
       
        ## Convolutional Layer Outputs - Calculations:
        #conv1
        out1=floor((img_dim+2*conv1_padding-conv1_kernel)/conv1_stride)+1
        #avg_pool1
        out2=floor((out1-pool1_kernel)/pool1_stride)+1
        #conv2
        out3=floor((out2+2*conv2_padding-conv2_kernel)/conv2_stride)+1
        #avg_pool2
        out4=floor((out3-pool2_kernel)/pool2_stride)+1
        #conv3
        out5=floor((out4+2*conv3_padding-conv3_kernel)/conv3_stride)+1
        #avg_pool3
        out6=floor((out5-pool3_kernel)/pool3_stride)+1
        out=out6
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=conv3_output*out*out , out_features=fc1_output)
        self.fc2 = nn.Linear(in_features=fc1_output, out_features=fc2_output)
        self.fc3 = nn.Linear(in_features=fc2_output, out_features=num_classes)
        
        # Define the activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # Define the batch normalization layer
        self.batchnorm = nn.BatchNorm2d(num_features=num_channels,momentum=bn_momentum)
        
        # Print_option
        self.print_option=print_option
        
        # Define Convolutional Layers Output Channels & Dimension
        self.out_channels=conv3_output
        self.out_dim=out
        
    def forward(self, x):
        x0=x
        # Pass the input through the convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x1=x
        x = self.avgpool1(x)
        x2=x
        
        x = self.conv2(x)
        x = self.relu(x)
        x3=x
        x = self.avgpool2(x)
        x4=x
        
        x = self.conv3(x)
        x = self.relu(x)
        x5=x
        x = self.avgpool3(x)
        x6=x
        
        # Flatten the output of the convolutional layers
        x = x.view(-1, self.out_channels*self.out_dim*self.out_dim)
        x7=x
        
        # Pass the flattened output through the fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x8=x
        x = self.batchnorm(x)
        x9=x
        
        x = self.fc2(x)
        x = self.relu(x)
        x10=x
        
        
        x = self.fc3(x)
        x = self.softmax(x)
        
        if self.print_option==1:
            print("---BulbaNet---")
            print("Input : ",list(x0.shape))
            print("After Conv1 + Relu : ",list(x1.shape))
            print("After AvgPool1 : ",list(x2.shape))
            print("After Conv2 + Relu : ",list(x3.shape))
            print("After AvgPool2 : ",list(x4.shape))
            print("After Conv3 + Relu : ",list(x5.shape))
            print("After AvgPool3 : ",list(x6.shape))
            print("After Flattening : ",list(x7.shape))
            print("After Fully-connected1 + ReLu : ",list(x8.shape))
            print("After Batch-Normalization : ",list(x9.shape))
            print("After Fully-connected2 + ReLu : ",list(x10.shape))
            print("Output - After Fully-connected3 + Softmax : ",list(x.shape))
            self.print_option=0
            
        return x
    
    

class CharmaNet(nn.Module):
    def __init__(self, num_classes=8, num_channels=1, img_dim=40,
                 conv1_output=128,conv1_kernel=2,conv1_stride=1,conv1_padding=1,
                 conv2_output=256,conv2_kernel=2,conv2_stride=1,conv2_padding=1,
                 conv3_output=390,conv3_kernel=2,conv3_stride=1,conv3_padding=1,     
                 avgpool_kernel=2,avgpool_stride=2,
                 maxpool_kernel=2,maxpool_stride=2,
                 fc1_output=2048,fc2_output=2048,
                 bn_momentum=0.1, p_dropout=0.5,
                 print_option=0):
        super(CharmaNet, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=num_channels, 
                               out_channels=conv1_output, 
                               kernel_size=conv1_kernel, stride=conv1_stride, 
                               padding=conv1_padding)
        
        self.conv2 = nn.Conv2d(in_channels=conv1_output, 
                               out_channels=conv2_output, 
                               kernel_size=conv2_kernel, stride=conv2_stride, 
                               padding=conv2_padding)
        self.conv3 = nn.Conv2d(in_channels=conv2_output, 
                               out_channels=conv3_output, 
                               kernel_size=conv3_kernel, stride=conv3_stride, 
                               padding=conv3_padding)
        
        # Define average pooling layers & batch normalization layer
        self.batchnorm = nn.BatchNorm2d(num_features=num_channels,momentum=bn_momentum)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel, 
                                 stride=maxpool_stride)
        self.avgpool = nn.AvgPool2d(kernel_size=avgpool_kernel, 
                                     stride=avgpool_stride)

       
        ## Convolutional Layer Outputs - Calculations:
        #conv1
        out1=floor((img_dim+2*conv1_padding-conv1_kernel)/conv1_stride)+1
        #conv2
        out2=floor((out1+2*conv2_padding-conv2_kernel)/conv2_stride)+1
        #max_pool
        out3=floor((out2-maxpool_kernel)/maxpool_stride)+1
        #conv3
        out4=floor((out3+2*conv3_padding-conv3_kernel)/conv3_stride)+1
        #avg_pool
        out5=floor((out4-avgpool_kernel)/avgpool_stride)+1
        out=out5
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=conv3_output*out*out , out_features=fc1_output)
        self.fc2 = nn.Linear(in_features=fc1_output, out_features=fc2_output)
        self.fc3 = nn.Linear(in_features=fc2_output, out_features=num_classes)
        
        # Define the activation functions
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # Define the Dropout Layer
        self.dropout=nn.Dropout(p=p_dropout)
        
        # Print_option
        self.print_option=print_option
        
        # Define Convolutional Layers Output Channels & Dimension
        self.out_channels=conv3_output
        self.out_dim=out
        
    def forward(self, x):
        x0=x
        # Pass the input through the convolutional layers
        x = self.conv1(x)
        x = self.elu(x)
        x1=x
        x = self.batchnorm(x)
        x2=x
        
        x = self.conv2(x)
        x = self.elu(x)
        x3=x
        x = self.maxpool(x)
        x4=x
        
        x = self.conv3(x)
        x = self.elu(x)
        x5=x
        x = self.avgpool(x)
        x6=x
        
        # Flatten the output of the convolutional layers
        x = x.view(-1, self.out_channels*self.out_dim*self.out_dim)
        x7=x
        
        # Pass the flattened output through the fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x8=x
        
        x = self.dropout(x)
        x9=x
        
        x = self.fc2(x)
        x = self.relu(x)
        x10=x
        
        x = self.fc3(x)
        x = self.softmax(x)
        
        if self.print_option==1:
            print("---CharmaNet---")
            print("Input : ",list(x0.shape))
            print("After Conv1 + ELU : ",list(x1.shape))
            print("After Batch-Normalization : ",list(x2.shape))
            print("After Conv2 + ELU : ",list(x3.shape))
            print("After MaxPool : ",list(x4.shape))
            print("After Conv3 + ELU : ",list(x5.shape))
            print("After AvgPool : ",list(x6.shape))
            print("After Flattening : ",list(x7.shape))
            print("After Fully-connected1 + ReLu : ",list(x8.shape))
            print("After Dropout : ",list(x9.shape))
            print("After Fully-connected2 + ReLu : ",list(x10.shape))
            print("Output - After Fully-connected3 + Softmax : ",list(x.shape))
            self.print_option=0
            
        return x
    
    
    
class SquirtleNet(nn.Module):
    def __init__(self, num_classes=8, num_channels=1, img_dim=40,
                 conv1_output=128,conv1_kernel=2,conv1_stride=1,conv1_padding=1,
                 conv2_output=256,conv2_kernel=2,conv2_stride=1,conv2_padding=1,
                 conv3_output=390,conv3_kernel=2,conv3_stride=1,conv3_padding=1,     
                 pool1_kernel=2,pool1_stride=2,
                 pool2_kernel=2,pool2_stride=2,
                 fc1_output=2048,fc2_output=2048,
                 p_dropout1=0.5, p_dropout2=0.5,
                 print_option=0):
        super(SquirtleNet, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=num_channels, 
                               out_channels=conv1_output, 
                               kernel_size=conv1_kernel, stride=conv1_stride, 
                               padding=conv1_padding)
        
        self.conv2 = nn.Conv2d(in_channels=conv1_output, 
                               out_channels=conv2_output, 
                               kernel_size=conv2_kernel, stride=conv2_stride, 
                               padding=conv2_padding)
        self.conv3 = nn.Conv2d(in_channels=conv2_output, 
                               out_channels=conv3_output, 
                               kernel_size=conv3_kernel, stride=conv3_stride, 
                               padding=conv3_padding)
        
        # Define average pooling layers & batch normalization layer
        self.avgpool1 = nn.AvgPool2d(kernel_size=pool1_kernel, 
                                 stride=pool1_stride)
        self.avgpool2 = nn.AvgPool2d(kernel_size=pool2_kernel, 
                                     stride=pool2_stride)

       
        ## Convolutional Layer Outputs - Calculations:
        #conv1
        out1=floor((img_dim+2*conv1_padding-conv1_kernel)/conv1_stride)+1
        #conv2
        out2=floor((out1+2*conv2_padding-conv2_kernel)/conv2_stride)+1
        #avg_pool1
        out3=floor((out2-pool1_kernel)/pool1_stride)+1
        #conv3
        out4=floor((out3+2*conv3_padding-conv3_kernel)/conv3_stride)+1
        #avg_pool2
        out5=floor((out4-pool2_kernel)/pool2_stride)+1
        out=out5
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=conv3_output*out*out , out_features=fc1_output)
        self.fc2 = nn.Linear(in_features=fc1_output, out_features=fc2_output)
        self.fc3 = nn.Linear(in_features=fc2_output, out_features=num_classes)
        
        # Define the activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # Define the Dropout Layer
        self.dropout1=nn.Dropout(p=p_dropout1)
        self.dropout2=nn.Dropout(p=p_dropout2)
        
        # Print_option
        self.print_option=print_option
        
        # Define Convolutional Layers Output Channels & Dimension
        self.out_channels=conv3_output
        self.out_dim=out
        
    def forward(self, x):
        x0=x
        # Pass the input through the convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x1=x
        x = self.dropout1(x)
        x2=x
        
        x = self.conv2(x)
        x = self.relu(x)
        x3=x
        x = self.avgpool1(x)
        x4=x
        
        x = self.conv3(x)
        x = self.relu(x)
        x5=x
        x = self.avgpool2(x)
        x6=x
        
        # Flatten the output of the convolutional layers
        x = x.view(-1, self.out_channels*self.out_dim*self.out_dim)
        x7=x
        
        # Pass the flattened output through the fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x8=x
        
        x = self.dropout2(x)
        x9=x
        
        x = self.fc2(x)
        x = self.relu(x)
        x10=x
        
        x = self.fc3(x)
        x = self.softmax(x)
        
        if self.print_option==1:
            print("---SquirtleNet---")
            print("Input : ",list(x0.shape))
            print("After Conv1 + ELU : ",list(x1.shape))
            print("After Dropout1 : ",list(x2.shape))
            print("After Conv2 + ELU : ",list(x3.shape))
            print("After MaxPool : ",list(x4.shape))
            print("After Conv3 + ELU : ",list(x5.shape))
            print("After AvgPool : ",list(x6.shape))
            print("After Flattening : ",list(x7.shape))
            print("After Fully-connected1 + ReLu : ",list(x8.shape))
            print("After Dropout2 : ",list(x9.shape))
            print("After Fully-connected2 + ReLu : ",list(x10.shape))
            print("Output - After Fully-connected3 + Softmax : ",list(x.shape))
            self.print_option=0
            
        return x
    
    

#---------------------------------------------------------------------#

################## TRAIN & TEST ####################
#--------------------------------------------------#

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader,num_classes):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0 for i in range(num_classes)]
    class_total = [0 for i in range(num_classes)]
    per_class_acc=[]
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %d %%' % accuracy)
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print('Accuracy of %5s : %2d %%' % (
                str(i), class_acc))
        per_class_acc.append(class_acc)
    return accuracy , per_class_acc

            