# 参加算能高性能计算竞赛赢取大奖
## 竞赛说明
* 参赛者报名后，使用算能AI芯片指令集对Conv2d、Depthwise2d、Matmul、Softmax算子进行编程，在保证正确性的前提下，我们对参赛者提交代码的性能进行排名，奖励排名靠前的团队或个人。
## 考察范围
* 参赛者只需完成okkernel/device下的ok_device_conv2d_contest.c ok_device_depthwise_contest.c ok_device_matmul_contest.c ok_device_softmax_contest.c 中TODO部分的代码，将此4个文件提交至svn(svn地址和密码在参赛者报名成功后会发送至邮箱)，我们对参赛者提交代码的性能进行排名，奖励排名靠前的团队或个人。
## 竞赛规则
* 本次比赛Conv2d、Depthwise2d、Matmul、Softmax4个算子的实现和性能优化
* 每个算子有多组参数，每个算子的每组参数称为一个case，每个case独立计分。
* 参赛者只需使用OKKernel实现Device端的代码，提交时也只提交Device端的代码。
* 参赛者提交的代码编译不通过视为失败提交，总分计0分。
* 参赛者提交的代码运行时不能导致芯片Hang死等异常情况发生，否则视为失败提交，总分计0分。
* 每个case只要实现正确至少可获得1分，并进入该case的性能排名环节。
* 每个case的性能名次与得分：第一名6分，第二名5分，第三名4分，第四名3分，第五名2分，其他1分。
* 计算所有case的得分总和，作为总分。
* 一共有45个case，理论满分为6 × 45 = 270分。
## 互动
* 主办方每天会检验参赛者的现有提交，并将每个case的性能数据以文件形式保存至参赛者的svn。
* 主办方每天会在github更新每个case的最优性能，以及各奖项当前对应的总分。
## 奖励办法
* 以总分由高到低的顺序为参赛者排名。
* 总分为135分及以上的参赛者有资格获得前三等奖励：  
  一等奖（1人）：奖金50000元  
  二等奖（2人）：奖金30000元  
  三等奖（5人）：奖金10000元  
* 总分为27分及以上的参赛者有资格获得优秀奖（50人）：奖金1000元。
* 奖励不重叠，获得一、二、三等奖的参赛者不获得优秀奖。
* 如果出现总分相同的情况，以提交时间为标准再次排名，提交越早，排名越靠前。
  
## 开发环境配置
## 如何编写程序
* 阅读文档  
  bm1684contest clone后，用浏览器打开doc/index.html。  
  仔细阅读Introduction至Storage Modes,了解sophgo芯片结构和内存布局  
  About Function Names至Fixed Point Unary Functions，介绍了编程中所需的所有结构和函数声明,参赛者可结合demo的使用
## 联系我们
![图片说明](https://github.com/sophon-ai-algo/bm1684contest/blob/main/pic/contact_us.jpg)
  
