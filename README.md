## References

<br/>


## Getting Started

### Prerequisites

这次使用到的数据集是新闻图片，共有6类，其中训练集数据有13972张照片，测试数据集有4517张照片。数据包含两个子目录分别train与test，采用隐形字典形式读取数据集。
### Model Structure
使用resnet18进行迁移学习，修改了最后全连接层的参数，模型结构如下：  
>ResNet(  
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
  (relu): ReLU(inplace=True)  
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)  
  (layer1): Sequential(  
    (0): BasicBlock(  
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
    )  
    (1): BasicBlock(  
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
    )  
  )  
  (layer2): Sequential(  
    (0): BasicBlock(  
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)  
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (downsample): Sequential(  
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)  
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      )  
    )  
    (1): BasicBlock(  
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
    )  
  )  
  (layer3): Sequential(  
    (0): BasicBlock(  
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)  
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (downsample): Sequential(  
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)  
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      )  
    )  
    (1): BasicBlock(  
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
    )  
  )  
  (layer4): Sequential(  
    (0): BasicBlock(  
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)  
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (downsample): Sequential(  
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)  
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      )  
    )  
    (1): BasicBlock(  
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu): ReLU(inplace=True)  
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
    )  
  )  
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))  
  (fc): Sequential(  
    (0): Dropout(p=0.3, inplace=False)  
    (1): Linear(in_features=512, out_features=6, bias=True)  
  )  
)  
## Results  
预测结果如下：  
**Epoch 1/10**  
344it [17:19,  3.02s/it]  
Batch 344, Train loss:0.0389, Train acc:0.3482, Time: 1039.535296201706  
112it [04:14,  2.27s/it]  
Batch 112, Test loss:0.0359, Test acc:0.4183, Time :254.46853590011597  
**Epoch 2/10**  
343it [16:52,  2.95s/it]  
Batch 344, Train loss:0.0373, Train acc:0.3759, Time: 1014.8161301612854  
344it [16:54,  2.95s/it]  
112it [04:11,  2.24s/it]  
Batch 112, Test loss:0.0363, Test acc:0.4167, Time :251.06466388702393  
**Epoch 3/10**  
343it [16:53,  2.96s/it]  
Batch 344, Train loss:0.0370, Train acc:0.3805, Time: 1015.6094932556152  
344it [16:55,  2.95s/it]  
112it [04:11,  2.24s/it]  
Batch 112, Test loss:0.0369, Test acc:0.4022, Time :251.03945183753967  
**Epoch 4/10**  
344it [16:55,  2.95s/it]  
0it [00:00, ?it/s]  
Batch 344, Train loss:0.0368, Train acc:0.3793, Time: 1015.3265600204468  
112it [04:10,  2.24s/it]  
Batch 112, Test loss:0.0360, Test acc:0.4230, Time :250.96127009391785  
0it [00:00, ?it/s]  
**Epoch 5/10**    
343it [16:53,  2.95s/it]  
Batch 344, Train loss:0.0366, Train acc:0.3815, Time: 1015.4724538326263  
344it [16:55,  2.95s/it]  
112it [04:10,  2.24s/it]  
Batch 112, Test loss:0.0361, Test acc:0.4121, Time :250.981507062912  
0it [00:00, ?it/s]  
**Epoch 6/10**    
344it [16:55,  2.95s/it]  
0it [00:00, ?it/s]  
Batch 344, Train loss:0.0367, Train acc:0.3848, Time: 1015.985337972641  
112it [04:11,  2.24s/it]  
Batch 112, Test loss:0.0358, Test acc:0.4357, Time :251.0967607498169  
0it [00:00, ?it/s]  
**Epoch 7/10**    
344it [16:56,  2.95s/it]  
Batch 344, Train loss:0.0365, Train acc:0.3842, Time: 1016.375559091568  
112it [04:11,  2.24s/it]  
Batch 112, Test loss:0.0359, Test acc:0.4315, Time :251.2618911266327  
**Epoch 8/10**  
344it [16:55,  2.95s/it]  
0it [00:00, ?it/s]  
Batch 344, Train loss:0.0368, Train acc:0.3821, Time: 1015.8622109889984  
112it [04:11,  2.24s/it]  
Batch 112, Test loss:0.0354, Test acc:0.4333, Time :251.07493114471436  
**Epoch 9/10**  
344it [16:56,  2.95s/it]  
Batch 344, Train loss:0.0367, Train acc:0.3826, Time: 1016.1267077922821  
112it [04:11,  2.24s/it]  
Batch 112, Test loss:0.0363, Test acc:0.4022, Time :251.1050910949707  
0it [00:00, ?it/s]  
**Epoch 10/10**    
344it [16:56,  2.96s/it]  
Batch 344, Train loss:0.0363, Train acc:0.3955, Time: 1016.6961767673492  
111it [04:10,  2.25s/it]  
Batch 112, Test loss:0.0359, Test acc:0.4223, Time :251.74002885818481  
112it [04:11,  2.25s/it]  
