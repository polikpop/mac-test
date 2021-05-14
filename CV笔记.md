### 图像分类

#### LESSON 4

- 看mixup代码，了解`batch_size` 和`index` 被赋值时的那两个操作。

```python
x[index, :]
```

看了之后还是比较疑惑这个操作。（5.10）

<img src="/Users/shuoxu/Library/Application Support/typora-user-images/截屏2021-05-09 上午12.25.55.png" alt="截屏2021-05-09 上午12.25.55" style="zoom:50%;" />

在train函数里面才得到了一个个batch(从dataloader中取出来的)，所以可以在这里使用mixup。

<img src="/Users/shuoxu/Library/Application Support/typora-user-images/截屏2021-05-09 上午12.37.19.png" alt="截屏2021-05-09 上午12.37.19" style="zoom:50%;" />

mixup在train中的代码，可以看到是否启用mixup也设置在了cfg中。

---



- python中的bool类型和c中的bool类型的区别

python: 是 True 和 False

c: 是 true 和 false



---



注意：打断点的时候要注意你打断点的这个地方会不会被调用，假如你跑的是当前的python脚本，很可能其实没有被调用。

---



- torch.long类型？pythorch的交叉熵？

torch.long 类型在官网描述中应该是等价于 torch.int64的。应该是64位整形数据。

因为pytorch自带的交叉熵要获得一个类再给你转化成one hot向量，而做了标签平滑后我们得到的就是向量，没法直接用pytorch自带的crossentropy，所以要自己重写。

---



这是自己定义的损失函数：

![截屏2021-05-09 上午9.13.45](/Users/shuoxu/Library/Application Support/typora-user-images/截屏2021-05-09 上午9.13.45.png)



smoothing表示要把标签的多少置信度拿出去做平滑。

- python中类定义时的super
- scatter_函数

- 为什么最后求和取平均 

  求和求的是102个和，取平均是1个batch64个取平均。最终loss要是标量。



<img src="/Users/shuoxu/Library/Application Support/typora-user-images/截屏2021-05-09 上午9.56.50.png" alt="截屏2021-05-09 上午9.56.50" style="zoom:50%;" />

想看代码中的某一段的运行结果可以这样，然后再看a的属性什么的。



---



#### LESSON 5

- RandomSampler官方doc

  为什么要输入每个样本的概率而不是每个类的概率。*未解决*

- python3函数注解

  在函数的参数后面加冒号然后补充这个参数希望的类型，这个注解是给使用者看的，实际上不会对运行产生什么影响。     *解决，已经录入自学手册*

- ppt中采样部分的sampler的github。
- 终极目标：看渐进式采样论文



---

渐进式采样：样本量分布从不均衡到均衡

![截屏2021-05-11 上午9.55.35](/Users/shuoxu/Library/Application Support/typora-user-images/截屏2021-05-11 上午9.55.35.png)

![截屏2021-05-11 上午10.38.12](/Users/shuoxu/Library/Application Support/typora-user-images/截屏2021-05-11 上午10.38.12.png)

q如果为0，那么可以得到每类样本的采样概率都为1.q如果为1，那么样本越多的类采样越多。初始时 t / T = 0， PIB最大，此时保持了原始分布采样，随着epoch的迭代，PCB逐渐变大，采样率逐渐变为均衡。

---

为了实现PB Sampling，我们要利用好Dataloader和Dataset。

- 要看一下cifar10如何解析数据集成为png图像。   *未解决，目前先看cifar10数据集组成*


<img src="/Users/shuoxu/Library/Application Support/typora-user-images/截屏2021-05-13 下午8.14.21.png" alt="截屏2021-05-13 下午8.14.21" style="zoom:50%;" />

在CIFAR-10 数据集中，文件data_batch_1.bin、data_batch_2.bin 、··data_batch_5.bin 和test_ batch.bin 中各有10000 个样本。一个样本由3073 个字节组成，第一个字节为标签label ，剩下3072 个字节为图像数据。样本和样本之间没高多余的字节分割， 因此这几个二进制文件的大小都是30730000 字节。

**重点就是怎么把二进制数据转化为标签和图片**



---



- 要看一下cifar10dataset函数的实现    *已解决*

学到了比较多的python函数，自己要多看看复习。

---



<img src="/Users/shuoxu/Library/Application Support/typora-user-images/截屏2021-05-11 上午11.04.51.png" alt="截屏2021-05-11 上午11.04.51" style="zoom:150%;" />

上图是get_img_info的实现。

由于img_info中已经存下了图片的数据，那么我们想把这个比较均衡的数据集变换成长尾分布其实只要对img_info做一些操作就可以了。所以我们会再写一个长尾分布的cifar10的dataset类，它继承自原来的**cifar10dataset**，主要是实现了对**dataset**的**img_info**的长尾变换。

新的类中主要实现了两个功能，一个是**把每个类的数目整成一个长尾分布的形式**。其次一个是**根据得到的数目把对应的图片从数据集中挑选出来**。![截屏2021-05-13 上午10.46.31](/Users/shuoxu/Library/Application Support/typora-user-images/截屏2021-05-13 上午10.46.31.png)

上图是长尾分布子类的具体实现，首先通过super初始化了父类的init，然后赋值了长尾分布的**imb_factor**，表示**最少的类别数目是最多的类别数目的多少**。然后判断是否在训练状态来决定是否要按长尾分配数量。

---

**PBsampling解析**

首先要明白我们要在哪里去改epoch中的样本分布，当然是在train时每次读出epoch时来重新构建dataloader来实现这一功能。![截屏2021-05-13 上午11.11.06](/Users/shuoxu/Library/Application Support/typora-user-images/截屏2021-05-13 上午11.11.06.png)

要注意，这个代码中的sampler_generator是一个类实例而不是一个函数。在这里具体生成了一个sampler。

- 了解python中类的`__call__()`方法

 ![截屏2021-05-13 下午1.06.23](/Users/shuoxu/Library/Application Support/typora-user-images/截屏2021-05-13 下午1.06.23.png)

上图是pb类的call函数。

可以发现首先是在计算pb的概率。接下来放上calpb函数的代码：

![截屏2021-05-13 下午1.08.36](/Users/shuoxu/Library/Application Support/typora-user-images/截屏2021-05-13 下午1.08.36.png)

下面是calclass的代码：

![截屏2021-05-13 下午1.29.09](/Users/shuoxu/Library/Application Support/typora-user-images/截屏2021-05-13 下午1.29.09.png)

这段代码其实就是复现公式，可以自己看看。

- python的map     *已解决*

- python的lambda   *已解决*

注意上述代码中p_pb最后还要除以对应类样本总数，原因是sampler是按照每一个样本分配权重的。

再回到call函数，先把array的pb转换成tensor，然后生成smapler返回。

![截屏2021-05-13 下午4.38.38](/Users/shuoxu/Library/Application Support/typora-user-images/截屏2021-05-13 下午4.38.38.png)

上面这行代码会把samples_weights搞成一个和train_targets一样大的列表，把train_targets的元素值当索引在p_pb中找。

---



![截屏2021-05-13 下午4.48.44](/Users/shuoxu/Library/Application Support/typora-user-images/截屏2021-05-13 下午4.48.44.png)

---

**记得看看思维导图**

---

