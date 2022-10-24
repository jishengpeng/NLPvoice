# NLPvoice
The process of learning phonetics

### 10.19
[新增语音信号基础知识整理](https://github.com/jishengpeng/NLPvoice/blob/main/NlpVoice/signal%20processing/Readme.md)
### 10.20
[新增语音信号处理的代码解析](https://github.com/jishengpeng/NLPvoice/tree/main/NlpVoice/signal%20processing/code)
###### 主要有三段代码，第一段代码是最原始的理解，并且将语音文件转化成短时傅里叶谱，第二段是将语音文件原始转化为mel谱，第三段是将tacotron模型的数据处理拿过来，跑通得到一种比较好的数据预处理得到mel谱的方式

### 10.21~10.22
[阅读语音合成书籍](https://github.com/jishengpeng/NLPvoice/blob/main/text_to_speech.pdf)


### 10.24
#### 1.阅读了Tacotron的论文
###### tacotron是开创了语音合成端到端系列的文章，首先输入是字符，输出是频谱（语谱）。然后再用一个简单的声码器将语谱转化为波，主体框架是seq2seq。编码器和解码器还是比较复杂的，用到了很多的trick和经典的架构，里面有很多细节。然后实验方面做的相对简单。
#### 2.简略阅读了Tacotron2的论文
###### Tacotron2使用了一个和优化后的Wavenet模型来代替Griffin-Lim算法（声码器部分），然后主要用的是梅尔频谱，同时也对Tacotron模型的一些细节也做了更改，不使用CBHG，而是使用普通的LSTM和Convolution layer，decoder每一步只生成一个frame，增加post-net，即一个5层CNN来精调mel-spectrogram最终生成了十分接近人类声音的波形。



#### 后续
1.继续啃书《语音合成》
2.B站视频语音合成框架
3.语音合成简单框架代码整理
4.阅读tacotron,tacotron2论文
5.整理tacotron2代码
6.list
