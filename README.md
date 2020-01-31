# SeisCompletionGAN
构建一种对抗生成网络对有缺失的地震二维图像进行重建恢复。

## 代码环境
python 3.5.4 / Tensorflow 1.9.0

## 训练集
使用真实地震工区的地震segy格式数据，通过matlab进行读取，分割，得到了10万二维地震图片作为训练数据，5000作为测试数据。

## 预处理
将.mat数据通过Python转换为图片，并采用缺失道的处理，在输入网络前进行[白化](https://blog.csdn.net/yyhhlancelot/article/details/92981855)处理加速网络的收敛。

## 生成器/补全网络
将预处理完成的数据输入补全网络。补全网络如下<br>
![com_net.jpg](https://github.com/yyhhlancelot/SeisCompletionGAN/blob/master/pics/com_net.jpg)
* 采用了白化操作——将输入数据分布变换到0均值，单位方差的分布，加速神经网络的收敛。<br>
* 补全网络采用了一种编码-解码的结构，类似于自编码器，从最初为了之后的处理而降低分辨率，这样的操作可以降低存储空间和计算时间。<br>
* 补全网络通过反卷积操作可以使图像在最初的降维后恢复原始的分辨率。<br>
* 补全网络未采用池化操作来降低分辨率，只使用带步长的卷积操作来降低图像分辨率，相比池化能够在缺失区域产生不那么模糊的纹理。<br>
* 补全网络采用均方误差作为生成器的损失。<br>
* 使用[BatchNormalization](https://blog.csdn.net/yyhhlancelot/article/details/92981855)加速模型的收敛，在一定程度缓解了梯度消失的问题。<br>
* 激活函数采用[ReLU](https://blog.csdn.net/yyhhlancelot/article/details/100304974)解决了zero-centered的问题，在正区间解决了sigmoid和tanh梯度消失的问题，同时在梯度下降时收敛远快于simoidh和tanh。<br>
* 补全网络使用了空洞卷积（dilated convolutional layers）。在整个网络结构中这是非常重要的一步，通过对低分辨率的图像使用空洞卷积，可以理解为模型可以更有效地看到更加广阔的区域，可以理解为感受野进一步增大，能够在不通过pooling损失信息的情况下也拥有较大的感受野。<br>
* 补全网络通过[反卷积](https://blog.csdn.net/yyhhlancelot/article/details/82983987)upsampling还原到原始图片的尺寸。<br>

## 判别器/判别网络
将补全网络输出的图像输入判别网络，判别网络会产生一个概率来判定该图像时真实图像还是通过补全网络产生的图像。判别网络如下<br>
![dis_net.jpg](https://github.com/yyhhlancelot/SeisCompletionGAN/blob/master/pics/dis_net.jpg)

* 判别器不采用任何池化层，用带有步长的卷积来达到降低分辨率的作用，同时进一步提取特征。<br>
* 判别器最后sigmoid函数对最后的特征向量生成一个概率，来判定是真实图像还是通过补全网络生成的图像。<br>
* 判别器采用交叉熵损失。<br>
* 整体损失由生成器的均方误差和判别器的交叉熵损失构成。<br>

## 实验效果
原始数据<br>
![ori.jpg](https://github.com/yyhhlancelot/SeisCompletionGAN/blob/master/pics/ori.jpg)<br>
缺失数据<br>
![incomplete.jpg](https://github.com/yyhhlancelot/SeisCompletionGAN/blob/master/pics/incomplete.jpg)<br>
重建数据<br>
![recon.jpg](https://github.com/yyhhlancelot/SeisCompletionGAN/blob/master/pics/recon.jpg)<br>

## 升级
未来考虑对网络结构进行升级，拟使用使用三维地震数据进行训练，构建张量重构网络。三维地震数据的恢复能够更加贴合实际应用场景。

## 参考文献
* Goodfellow, Ian J , et al. "Generative Adversarial Nets." International Conference on Neural Information Processing Systems MIT Press, 2014.
* Chang, J. H. Rick , et al. "One Network to Solve Them All --- Solving Linear Inverse Problems using Deep Projection Models." (2017).
* Iizuka, Satoshi , E. Simo-Serra , and H. Ishikawa . "Globally and locally consistent image completion." ACM Transactions on Graphics 36.4(2017):1-14.
* Ioffe, Sergey , and C. Szegedy . "Batch normalization: accelerating deep network training by reducing internal covariate shift." International Conference on International Conference on Machine Learning JMLR.org, 2015.
