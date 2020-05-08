*本文为博主学习机器学习决策树部分的一些笔记和思考，以及python编程实现算法的具体步骤*

**决策树(decision tree)** 是一类常见的机器学习方法. 在已知各种情况发生概率的基础上，通过构成决策树来求取净现值的**期望值**大于等于零的概率，评价项目风险，判断其可行性的决策分析方法，是直观运用概率分析的一种**图解法**. 在机器学习中，决策树是一个**预测模型**，他代表的是对象属性与对象值之间的一种**映射关系**.

对于决策树的原理和概念以及信息增益的计算部分不再赘述，对此部分感兴趣或者希望了解的朋友可以翻阅周志华《机器学习》P73～P78

#### 本文重点介绍CART决策树
## 一、基本概念
CART决策树[Breiman et al., 1984] 使用“基尼指数”来选择划分属性. 这是西瓜书上给出的定义. 通过大量文章的阅读将CART决策树关键点整理如下：

 1. CART决策树既能是分类树，也能是回归树
 2. 当CART是分类树时，采用GINI值作为节点分裂的依据；当CART是回归树时，采用样本的最小方差作为节点分裂的依据
 3. ==CART是一颗二叉树== (关于这一点其实我存在疑惑，准备去问问老师或者同学。因为在我编程实现的过程中忽略了这一点，因为西瓜树上并没有指明CART算法必须生成二叉树. 而其中西瓜数据集中的离散属性取值$N\geq3$，因此我在编程过程中生成的是多叉树. 所以是否考虑在属性取值较少的情况下，CART算法不用一定生成二叉树)


## 二、选择划分属性
目标取值为一个有限集合的树模型称为**分类树**，而目标值是连续值(典型的真实数值)的树模型称为**回归树**。分裂的目的是为了能够让数据变纯，使决策树输出的结果更接近真实值。那么CART是如何评价节点的纯度呢？如果是分类树，CART采用GINI值衡量节点纯度；如果是回归树，采用样本方差衡量节点纯度. 节点越不纯，节点分类或者预测的效果就越差.
#### 1、CART决策树作为分类树 
CART决策树作为分类树时，特征属性可以是连续类型也可以是离散类型，但观察属性(即标签属性或者分类属性)必须是离散类型。

划分的目的是为了能够让数据变纯，使决策树输出的结果更接近真实值。

如果是**分类树**，CART决策树使用“基尼指数”来选择划分属性，数据集D的纯度可以用基尼值来度量：
$$Gini(D)=\sum_{k=1}^{\vert \mathcal Y\vert}\sum_{k'\neq k}p_kp_{k'}$$

$$=1-\sum_{k=1}^{\vert \mathcal Y\vert}p_k^2  $$

直观来说，$Gini(D)$反映了从数据集$D$中随机抽取两个样本，其类别标记不一致的概率。因此，$Gini(D)$越小，则数据集$D$的纯度越高. 

属性$a$的基尼指数定义为
$$Gini\_ index(D,a)={\sum_{v=1}^V \frac{\vert D^v\vert}{\vert D\vert}}Gini(D^v).$$
所以我们在候选属性集合A中，选择那个使得划分后基尼指数最小的属性作为最优划分属性，即$a_*={arg}_{a \in A}min\ Gini\_ index(D,a)$.

#### 2、CART作为回归树
而如果是**回归树**，其建立算法同CART分类树大部分是类似的。除了上述中提到的两者样本输出的不同，CART回归树和CART分类树的建立和预测主要有以下两点不同：

- 连续值的处理方法不同
- 决策树建立后做预测的方式不同

回归树对于连续值的处理使用了常见的和方差的度量方式，即：
$$\sigma=\sqrt {\sum_{i\in I}(x_i-\mu )^2}=\sqrt {\sum_{i\in I}{x_i}^2-n\mu ^2}.$$
方差越大，表示该节点的数据越分散，预测的效果就越差。如果一个节点的所有数据都相同，那么方差就为0，此时可以很肯定得认为该节点的输出值；如果节点的数据相差很大，那么输出的值有很大的可能与实际值相差较大。

**因此，无论是分类树还是回归树，CART都要选择使子节点的GINI值或者回归方差最小的属性作为分裂的方案。**

即最小化分类树:
$$Gain=\sum_{i\in I}p_i\cdot Gini_i  $$
或者回归树：
$$Gain=\sum _{i\in I}\ \sigma _i.$$

## 三、剪枝处理
剪枝(pruning)是决策树学习算法对付“过拟合”的主要手段. 决策树剪枝的基本策略有“**预剪枝**”(prepruning)和“**后剪枝**”(post-pruning). 预剪枝是指在决策树生成过程中，对每个结点在划分前先进行估计，若当前结点的划分不能带来决策树泛化性能提升，则停止划分并将当前结点标记为叶结点;   后剪枝则是先从训练集生成一颗完整的决策树，然后自底向上地对非叶结点进行考察，若将该结点对应的子树替换为叶结点能带来决策树泛化性能提升，则将该子树替换为叶结点.

**预剪枝**使得决策树很多分支都没有“展开”，这不仅降低了过拟合的风险，还显著减少了决策树的训练时间开销和测试时间开销. 但另一方面，预剪枝由于基于“贪心”本质将某些禁止展开的分支的后续划分也一并禁止，而其中某些后续划分有可能会导致泛化性能显著提高，给预剪枝决策树带来了欠拟合的风险. 

**后剪枝**决策树通常比预剪枝决策树保留了更多的分支，且在一般情况下欠拟合风险很小，泛化性能往往也能优于预剪枝决策树。但由于其是在生成完全决策树之后进行的，并且要自底向上考察，因此其训练时间开销比未剪枝决策树和预剪枝决策树都要大很多. 

## 四、连续值处理
由于现实学习任务中常会遇到连续属性，由于连续属性的可取值树木不再有限，此时**连续属性离散化技术**可派上用场. 最简单的策略是采用**二分法(bi-partion)**对连续属性进行处理，这正是C4.5决策树算法中采用的机制.

给定样本集合$D$和连续属性$a$，假定$a$在$D$上出现了n个不同的取值，将这些值从小到大进行排序，记为{$a^1,a^2,... ,a^n$}. 基于划分点$t$可将子集$D^-_t$和$D^+_t$，其中$D^-_t$包含那些在属性$a$上取值不大于$t$的样本，而$D^+_t$则包含那些在属性$a$上取值大于$t$的样本. 所以对连续属性$a$，我们可考察包含$n-1$个元素的候选划分点集合
$$T_a= \left\{ \frac {a^i=a^{i+1}}{2} \vert1 \leq i \leq n-1 \right\}$$
即把区间$\left[a^i,a^{i+1} \right)$的中位点$\frac{a^i+a^{i+1}}{2}$作为候选划分点. 然后我们就可以像离散属性值一样来考察这些划分点.

## 五、python代码实现

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

data_str = output = io.StringIO('''编号,色泽,根蒂,敲声,纹理,脐部,触感,密度,含糖率,好瓜
1,青绿,蜷缩,浊响,清晰,凹陷,硬滑,0.697,0.46,是
2,乌黑,蜷缩,沉闷,清晰,凹陷,硬滑,0.774,0.376,是
3,乌黑,蜷缩,浊响,清晰,凹陷,硬滑,0.634,0.264,是
4,青绿,蜷缩,沉闷,清晰,凹陷,硬滑,0.608,0.318,是
5,浅白,蜷缩,浊响,清晰,凹陷,硬滑,0.556,0.215,是
6,青绿,稍蜷,浊响,清晰,稍凹,软粘,0.403,0.237,是
7,乌黑,稍蜷,浊响,稍糊,稍凹,软粘,0.481,0.149,是
8,乌黑,稍蜷,浊响,清晰,稍凹,硬滑,0.437,0.211,是
9,乌黑,稍蜷,沉闷,稍糊,稍凹,硬滑,0.666,0.091,否
10,青绿,硬挺,清脆,清晰,平坦,软粘,0.243,0.267,否
11,浅白,硬挺,清脆,模糊,平坦,硬滑,0.245,0.057,否
12,浅白,蜷缩,浊响,模糊,平坦,软粘,0.343,0.099,否
13,青绿,稍蜷,浊响,稍糊,凹陷,硬滑,0.639,0.161,否
14,浅白,稍蜷,沉闷,稍糊,凹陷,硬滑,0.657,0.198,否
15,乌黑,稍蜷,浊响,清晰,稍凹,软粘,0.36,0.37,否
16,浅白,蜷缩,浊响,模糊,平坦,硬滑,0.593,0.042,否
17,青绿,蜷缩,沉闷,稍糊,稍凹,硬滑,0.719,0.103,否''')

data = pd.read_csv(data_str)
data.set_index('编号', inplace=True)
print(data)

##初始化训练集，西瓜数据集3.0
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023010916945.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RhaXl1Y2hlbmc4OA==,size_16,color_FFFFFF,t_70)
```python
#计算系统信息熵（后来弃用）
def entropy(data):
    length = data.size
    ent = 0
    for i in data.value_counts(): #查看表格某列中有多少个不同值
        prob = i / length
        ent += - prob * (np.log2(prob))
    return ent
print('--------')
entD = entropy(data['好瓜'])
print(entD) # 0.9975025463691153

#计算Gini指数
```
分别计算离散属性和连续属性的基尼指数
```python
def gini_discrete(data, input_column, output_column): #离散
    """
    data: input the name of DataFrame
    input_column: the name of feature
    output_column: good or bad
    return: none
    """
    ret = 0
    lens = data[output_column].size
    all_attribute = data[input_column].value_counts()  # 保存特征全部属性的取值个数
    for name in data[input_column].unique(): # 特征的不同属性名
        print(name)
        temp = 1
        for i in range(len(data[output_column].unique())):  # 好瓜 or 坏瓜
            attribute_num = data[input_column].where(data[output_column] == data[output_column].unique()[i]).value_counts()
            try:
                prob = int(attribute_num[name]) / int(all_attribute[name])
            except:
                prob = 0
            if prob == 0:
                temp += 0
            else:
                temp -= prob * prob
            # 还需要乘以该属性出现的概率
        ret += temp * (all_attribute[name] / lens)
    return ret

def gini_continuous(data, input_column, output_column): #连续
    """
    data: input the name of DataFrame
    input_column: the name of feature
    output_column: good or bad
    return: none
    """
    lens = data[output_column].size
    gini = 0
    T = []
    Gini = [] #用来寻找最小的gini_index
    #采用二分法，参考周志华《机器学习》P84
    values = sorted(data[input_column].values)
    for i in range(lens - 1):
        good_n = 0
        good_p = 0
        bad_n = 0
        bad_p = 0
        t = round(((values[i] + values[i+1]) / 2), 3)
        T.append(t)
        for index in data.index:
            if data[input_column].values[index-1] < t:
                if data['好瓜'].values[index-1] == '是' :
                    good_n += 1
                else:
                    bad_n += 1
            else:
                if data['好瓜'].values[index-1] == '是' :
                    good_p += 1
                else:
                    bad_p += 1
        
        dn_sum = i + 1 #小于候选划分总和
        dp_sum = lens - i - 1 #大于候选划分总和
        prob = dn_sum / lens
        gini_n = 1 - (np.square(good_n / dn_sum) + np.square(bad_n / dn_sum))
        gini_p = 1 - (np.square(good_p / dp_sum) + np.square(bad_p / dp_sum))
        gini = prob * gini_n + (1 - prob) * gini_p
        Gini.append(gini)
        
    print("对应划分点为：",T[Gini.index(min(Gini))])
    return T[Gini.index(min(Gini))], min(Gini)
```
根据基尼指数选择最佳划分点
```python
def chooseBestFeature(data):
    gini_min = 999
    at = 0 #at为连续值划分的最佳划分点
    for name in data.columns[:-1]:
        print('属性:', name)
        print('全部取值：')
        if name != '密度' and name != '含糖率' :
            gini = gini_discrete(data, name, '好瓜')
        else:
            at, gini = gini_continuous(data, name, '好瓜')
        print(name)
        if gini < gini_min:
            gini_min = gini
            name_min = name
            at_min = at
        print(gini)
        print()
    #print(entD)
    print('最小基尼指数：', gini_min, "属性名称：", name_min)
    print('所以应该基于{}划分'.format(name_min))
    v = data.columns.values
    v = v.tolist()
    bestFeatureIndex = v.index(name_min)
    print(bestFeatureIndex)
    return bestFeatureIndex, name_min, at_min
```
```python
max_n_features = 4 #控制树的深度

decisionTree = {}

def transLabel(label):
    if label == '是':
        return '好瓜'
    else:
        return '坏瓜'

def createTree(data, features):
    """
    data: input the name of DataFrame
    features: input the list of features
    """
    bestFeaIndex, bestFeatureName, at = chooseBestFeature(data) 
    bestFeatureValue = data[bestFeatureName].values
    attrCount = 0
    attr = []
    
    sameLvTree = {}
    tempTree = {}
    
    DataGood = pd.DataFrame()
    DataBad = pd.DataFrame()
    
    features.remove(bestFeatureName)
    
    if bestFeatureName != '密度' and '含糖率':      
        for name in data[bestFeatureName].unique():
            attrCount += 1
            attr.append(name)
        for i in range(attrCount):
            dataName='DataSubSet'+str(i) #根据属性的取值个数动态生成子集
            locals()['DataSubSet'+str(i)] = pd.DataFrame()
        for index in data.index:
            for i in range(attrCount):
                if data[bestFeatureName].values[index-1] == attr[i]:
                    locals()['DataSubSet'+str(i)] = locals()['DataSubSet'+str(i)].append(data[index-1:index], sort=False)
                    break
        
        outputCount = 0
        for i in range(attrCount):
            print()
            print(attr[i])
            locals()['DataSubSet'+str(i)] = locals()['DataSubSet'+str(i)].drop(columns = [bestFeatureName])
            locals()['DataSubSet'+str(i)] = locals()['DataSubSet'+str(i)].reset_index(drop=True)
            locals()['DataSubSet'+str(i)].index += 1
            print(locals()['DataSubSet'+str(i)])
            print()
            for name in locals()['DataSubSet'+str(i)]['好瓜'].unique():
                outputCount += 1
            if outputCount == 1:
                print("*******",bestFeatureName, attr[i], locals()['DataSubSet'+str(i)]['好瓜'].values[0])
                sameLvTree[attr[i]] = transLabel(locals()['DataSubSet'+str(i)]['好瓜'].values[0])
                tempTree[bestFeatureName] = sameLvTree
                outputCount = 0
            else:
                print("*******",bestFeatureName, attr[i], '?')
                outputCount = 0
                if len(features) > max_n_features:
                    sameLvTree[attr[i]] = createTree(locals()['DataSubSet'+str(i)], features)
                    print(sameLvTree[attr[i]])
                    tempTree[bestFeatureName] = sameLvTree
    
    else:
        
        DataN = pd.DataFrame()
        DataP = pd.DataFrame()
        for index in data.index:
            if data[bestFeatureName].values[index-1] < at:
                DataN = DataN.append(data[index-1:index], sort=False)
            else:
                DataP = DataP.append(data[index-1:index], sort=False)
        
        outputCount = 0
        
        print()
        print('<=', at)
        DataN = DataN.drop(columns = [bestFeatureName])
        DataN = DataN.reset_index(drop=True)
        DataN.index += 1
        print(DataN)
        print()
        for name in DataN['好瓜'].unique():
            outputCount += 1
        if outputCount == 1:
            print("*******",bestFeatureName, '<={}'.format(at), DataN['好瓜'].values[0])
            sameLvTree['<={}'.format(at)] = transLabel(DataN['好瓜'].values[0])
            tempTree[bestFeatureName] = sameLvTree
            print(tempTree)
            outputCount = 0
        else:
            print("*******",bestFeatureName, '<={}'.format(at), '?')
            outputCount = 0
            if len(features) > max_n_features:
                sameLvTree['<={}'.format(at)] = createTree(DataN, features)
                tempTree[bestFeatureName] = sameLvTree
                print(tempTree)
        
        print()
        print('>', at)
        DataP = DataP.drop(columns = [bestFeatureName])
        DataP = DataP.reset_index(drop=True)
        DataP.index += 1
        print(DataP)
        print()
        for name in DataP['好瓜'].unique():
            outputCount += 1
        if outputCount == 1:
            print("*******",bestFeatureName, '>{}'.format(at), DataP['好瓜'].values[0])
            print(bestFeatureName)
            sameLvTree['>{}'.format(at)] = transLabel(DataP['好瓜'].values[0])
            tempTree[bestFeatureName] = sameLvTree
            outputCount = 0
        else:
            print("*******",bestFeatureName, '>{}'.format(at), '?')
            outputCount = 0
            if len(features) > max_n_features:
                sameLvTree['>{}'.format(at)] = createTree(DataP, features)
                tempTree[bestFeatureName] = sameLvTree
        
    return tempTree
```
```python
features = list(data.columns[0:-1]) # x的表头
decisionTree = createTree(data, features)
print(decisionTree)
```
下面的代码就是直接借鉴的别人的代码，能够根据固定形式的字典生成决策树，参考[python绘制决策树](https://blog.csdn.net/sinat_29957455/article/details/76553987)
```python
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

def getNumLeafs(myTree):
    #初始化树的叶子节点个数
    numLeafs = 0
    #myTree.keys()获取树的非叶子节点'no surfacing'和'flippers'
    #list(myTree.keys())[0]获取第一个键名'no surfacing'
    firstStr = list(myTree.keys())[0]
    #通过键名获取与之对应的值，即{0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}
    secondDict = myTree[firstStr]
    #遍历树，secondDict.keys()获取所有的键
    for key in secondDict.keys():
        #判断键是否为字典，键名1和其值就组成了一个字典，如果是字典则通过递归继续遍历，寻找叶子节点
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        #如果不是字典，则叶子结点的数目就加1
        else:
            numLeafs += 1
    #返回叶子节点的数目
    return numLeafs

def getTreeDepth(myTree):
    #初始化树的深度
    maxDepth = 0
    #获取树的第一个键名
    firstStr = list(myTree.keys())[0]
    #获取键名所对应的值
    secondDict = myTree[firstStr]
    #遍历树
    for key in secondDict.keys():
        #如果获取的键是字典，树的深度加1
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        #去深度的最大值
        if thisDepth > maxDepth : maxDepth = thisDepth
    #返回树的深度
    return maxDepth


#设置画节点用的盒子的样式
decisionNode = dict(boxstyle = "sawtooth",fc="0.8")
leafNode = dict(boxstyle = "round4",fc="0.8")
#设置画箭头的样式    http://matplotlib.org/api/patches_api.html#matplotlib.patches.FancyArrowPatch
arrow_args = dict(arrowstyle="<-")
#绘图相关参数的设置
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    #annotate函数是为绘制图上指定的数据点xy添加一个nodeTxt注释
    #nodeTxt是给数据点xy添加一个注释，xy为数据点的开始绘制的坐标,位于节点的中间位置
    #xycoords设置指定点xy的坐标类型，xytext为注释的中间点坐标，textcoords设置注释点坐标样式
    #bbox设置装注释盒子的样式,arrowprops设置箭头的样式
    '''
    figure points:表示坐标原点在图的左下角的数据点
    figure pixels:表示坐标原点在图的左下角的像素点
    figure fraction：此时取值是小数，范围是([0,1],[0,1]),在图的左下角时xy是（0,0），最右上角是(1,1)
    其他位置是按相对图的宽高的比例取最小值
    axes points : 表示坐标原点在图中坐标的左下角的数据点
    axes pixels : 表示坐标原点在图中坐标的左下角的像素点
    axes fraction : 与figure fraction类似，只不过相对于图的位置改成是相对于坐标轴的位置
    '''
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,\
    xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',\
    va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)

#绘制线中间的文字(0和1)的绘制
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]   #计算文字的x坐标
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]   #计算文字的y坐标
    createPlot.ax1.text(xMid,yMid,txtString)
#绘制树
def plotTree(myTree,parentPt,nodeTxt):
    #获取树的叶子节点
    numLeafs = getNumLeafs(myTree)
    #获取树的深度
    depth = getTreeDepth(myTree)
    #firstStr = myTree.keys()[0]
    #获取第一个键名
    firstStr = list(myTree.keys())[0]
    #计算子节点的坐标
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,\
              plotTree.yoff)
    #绘制线上的文字
    plotMidText(cntrPt,parentPt,nodeTxt)
    #绘制节点
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    #获取第一个键值
    secondDict = myTree[firstStr]
    #计算节点y方向上的偏移量，根据树的深度
    plotTree.yoff = plotTree.yoff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            #递归绘制树
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            #更新x的偏移量,每个叶子结点x轴方向上的距离为 1/plotTree.totalW
            plotTree.xoff = plotTree.xoff + 1.0 / plotTree.totalW
            #绘制非叶子节点
            plotNode(secondDict[key],(plotTree.xoff,plotTree.yoff),\
                     cntrPt,leafNode)
            #绘制箭头上的标志
            plotMidText((plotTree.xoff,plotTree.yoff),cntrPt,str(key))
    plotTree.yoff = plotTree.yoff + 1.0 / plotTree.totalD
    
    

#绘制决策树，inTree的格式为{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}


def createPlot(inTree):
    #新建一个figure设置背景颜色为白色
    fig = plt.figure(1,facecolor='white')
    #清除figure
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    #创建一个1行1列1个figure，并把网格里面的第一个figure的Axes实例返回给ax1作为函数createPlot()
    #的属性，这个属性ax1相当于一个全局变量，可以给plotNode函数使用
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    #获取树的叶子节点
    plotTree.totalW = float(getNumLeafs(inTree))
    #获取树的深度
    plotTree.totalD = float(getTreeDepth(inTree))
    #节点的x轴的偏移量为-1/plotTree.totlaW/2,1为x轴的长度，除以2保证每一个节点的x轴之间的距离为1/plotTree.totlaW*2
    plotTree.xoff = -0.5/plotTree.totalW
    plotTree.yoff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()
```
```python
createPlot(decisionTree)
```
最终绘制决策树如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023011324297.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RhaXl1Y2hlbmc4OA==,size_16,color_FFFFFF,t_70)
*注意上述代码没有采取剪枝操作，因为数据集过小剪枝没有太大意义，感兴趣的读者可以采用更大的数据集来编写剪枝代码*
> 参考文章
> * 周志华《机器学习》
> * [决策树-CART回归树](http://blog.csdn.net/beauty0522/article/details/82726866)
> * [CART分类树原理及示例](http://blog.csdn.net/aaa_aaa1sdf/article/details/81587359)
> * [分类回归树](https://www.cnblogs.com/jin-liang/p/9706117.html)