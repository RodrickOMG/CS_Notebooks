## 第1节 视图

- 视图采用create view语句定义，
- 可以是任何一个SQL语句。
- 无实际数据的'虚表'，有利于数据一致性！
- 视图上可以在定义新的视图！

![图片7](https://tva1.sinaimg.cn/large/007S8ZIlly1gfixpcx6hpj30do09s74s.jpg)

```mysql
 create view faculty as 
 select ID, name, dept_name
 from instructor ;
```

1. 可以在任何QL语句中像表一样的被使用！
2. 增强查询能力且方便(用户/程序员)使用！
3. 还可以提供数据访问安全控制(隐藏数据)！
4. 作为外模式(1级映射)有利于应用独立性！

## 第2节 完整性约束

### 2.1 完整性约束

#### 2.1.1 键完整性约束（主码/主键）

- 关系(模式)必需有一个主码，来区分不同元组！

- SQL采用primary key…来定义！

#### 2.1.2 参照完整性约束（外码/外键）

- 用另一关系的主码，来约束属性取值的有效性！
- SQL采用foreign key … references …来定义！

#### 2.1.3 其它数据完整性约束

![图片8](https://tva1.sinaimg.cn/large/007S8ZIlly1gfiy69pbruj30pi06ojuz.jpg)

![图片9](https://tva1.sinaimg.cn/large/007S8ZIlly1gfiy79ojmvj312k09ejw8.jpg)

### 2.2 外键约束方式

![图片10](https://tva1.sinaimg.cn/large/007S8ZIlly1gfiya7xvapj30hc06iabh.jpg)

### 2.3 断言

例子：约束要求：student每个元组的tot_cred(学生的总学分)取值应等于该生所修完课程的学分总和(关系takes∞course的credits)

![图片11](https://tva1.sinaimg.cn/large/007S8ZIlly1gfiybhc1brj30pa09q780.jpg)

## 第3节 授权

### 3.1 表（关系）上的授权

#### SQL如何限制用户对表中数据的合法访问？ 

通过授权！只有授权用户才能查看(/插入/修改/删除)相关表中的数据。注:表的创建者, 自然拥有表上的一切权限.![图片24](https://tva1.sinaimg.cn/large/007S8ZIlly1gfjt6n1husj30xq0ng13t.jpg)

### 3.2 视图上的授权

- 在表instructor上创建一个视图geo_instructor 

```mysql
create view geo_instructor as
(select *
 from instructor 
 where dept_name='Geology');
```

- 将视图上的查看权授予一个角色geo_staff

```mysql
grant select on geo_instructor to  geo_staff;
```

- 如果该用户在instructor上没有获得select授权，则他仍然看不到数据！（注：视图上的update权限也类似）

```mysql
select * from geo_instructor;
```

- 用户可以定义函数与过程(p.83,&5.2)，并可对其他用户授予execute执行权。

- **注：**

  1. 视图的创建者，自然拥有该视图上的所有权限！

  2. 函数与过程的创建者，自然拥有其上所有权限！
  3. 从SQL2003标准开始，允许在定义函数和过程时指定sql security invoker，执行时与调用者相同的权限下运行！

### 3.3 授权图（p84）

指定权限从一个用户到另一个用户的传递可以表示为授权图。该图中的顶点是用户。

#### 作用：

1. 描述在一张表上某种授权的当前状态，便于系统动态管理授权；
2. 当DBA或具有权限的用户(树上节点)进行授权时，树扩展(生长)； 
3. 当DBA或具有权限的用户(树上节点)回收权限时，树收缩(枯萎)；

#### 用户具有权限的充分必要条件是：

当且仅当存在从授权图的根（即代表数据库管理员的顶点）到代表该用户顶点的路径。

## 第4节 SQL存储过程

### 4.1 SQL函数

SQL除了提供一些常用的内建函数(聚集、日期、字符串转换等)外，可编写存储过程(业务逻辑)并存于库中, 可在SQL/应用代码中调用！

![图片25](https://tva1.sinaimg.cn/large/007S8ZIlly1gfjtsumnksj312y0panby.jpg)

### 4.2 SQL过程

![图片26](https://tva1.sinaimg.cn/large/007S8ZIlly1gfjtu27okij310u0ouaj0.jpg)

### 4.3 外部语言过程

- 外部语言过程：SQL允许用程序语言(Java,C#,C,C++)来定义函数或过程，
- 运行效率要高于SQL中定义的函数，用于完成无法在SQL中执行的计算。

## 第5节 触发器

**触发器**是一条语句，当对数据库作修改时，它自动被系统执行。要设置触发器机制，必须满足两个要求：

- 指明什么条件下执行触发器。它被分解为一个引起触发器被检测的事件和一个触发器执行必须满足的条件
- 指明触发器执行时的动作

### 对触发器的需求

触发器可以用来实现未被SQL约束机制指定的某些完整性约束。它还是一种非常有用的机制，用来当满足特定条件时对用户发警报或自动开始执行某项任务。触发器还可用于复制或备份数据库。

![图片27](https://tva1.sinaimg.cn/large/007S8ZIlly1gfjumqouu1j313x0u0e0f.jpg)

