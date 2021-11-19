## 第1节 SQL查询语言概览

- **数据定义语言（Data-Definition Language，DDL）**：SQL DDL提供定义关系模式、删除关系以及修改关系模式的命令。
- **数据操纵语言（Data-Manipulation Language，DML）：**SQL DML提供从数据库中查询信息，以及在数据库中插入元组、删除元组、修改元组的能力。
- **完整性**： SQL DDL包括定义完整性约束的命令，保存在数据库中的数据必须满足所定义的完整性约束。破坏完整性约束的更新是不被允许的。
- **视图定义：**SQL DDL包括定义视图的命令。
- **事务控制：**SQL包括定义事务的开始和结束的命令。
- 嵌入式SQL和动态SQL
- 授权(authorization)：SQL DDL包括定义对关系和视图的访问权限的命令。

### SQL如何实际建立一个关系模式结构？

```sql
create tabel department
( dept_name varchar(20),
  building varchar(15),
  budget numeric(12,2),
  primary key (dept_name));
```

###  为什么要定义完整性约束？

因为完整性约束保证授权用户对数据库所做的修改不会破坏数据的一致性，可以防止对数据的意外破坏。

### create database到底做了什么？

1. 产生了一个数据库（空仓库，仅包括系统数据字典）
2. 初始库小，数据增长需要时才增大库空间
3. 同时，还产生了一个日志存放的空仓库（备份回复用）
4. 还涉及到物理设计工作：库放在何位置、库大小、库增量。而且日志仓库位置可用与数据仓库位置不同(保证安全)！

## 第2节 SQL数据定义

数据库中的关系集合必须由数据定义语言（DDL）指定给系统。SQL的DDL不仅能够定义一组关系，还能够定义每个关系的信息。

### 3.2.1 基本类型

- char(n)：固定长度的字符串，用户指定长度n
- varchar(n)：可变长度的字符串，用户指定最大长度n
- int(n)：整数类型（和机器相关的相关的整数类型的子集）
- numeric(p, d)：定点数，精度由用户指定。这个数有p位数字（加上一个符号位），其中d位数字在小数点右边。
- real，double precision：浮点数与双精度浮点数，精度与机器相关。
- float(n)：精度至少位n位的浮点数

### 3.2.2 基本模式定义

```sql
create tabel department
( dept_name varchar(20),
  building varchar(15),
  budget numeric(12,2),
  primary key (dept_name));
```

SQL支持许多不同的完整性约束

- **primary key：** primary key声明表示属性Aj1，Aj2，…，Ajm构成关系的主码。主码属性必须非空且唯一，也就是说没有一个元组在主码属性上取空值，关系中也没有两个元组在主码属性上取值相同。
- **foreign key reference：** foreign key声明表示关系中任意元组在属性(Ak1, Ak2, ..., Akn)上的取值必须对应于关系s中某元组在主码属性上的取值。
- **not null：**一个属性上的not null约束表明在该属性上不允许空值。

## 第3节 SQL查询的基本结构

### 3.3.1 单关系查询

```sql
select name
from instructor;
```

删除重复

```sql
select distinct dept_name
from instructor;
```

SQL允许我们使用关键词all来显式指明不去除重复：

```mysql
select all dept_name
from instructor;
```

**where** 子句允许我们只选出那些在**from** 子句的结果关系中满足特定谓词的元组：

```mysql
select name
from instuctor
where depat_name = ‘Comp.Sci’ and salaty > 70000;
```

#### Where子语句在关系代数操作上的作用?

1. 关系记录的筛选
2. 两关系间的连接

#### 自然连接Natural join与迪卡儿积Χ两点最大不同

1. 仅包含符合连接条件的元组
2. 连接属性仅出现一次

### 3.3.2 多关系查询

```sql
select name, instructor.dept_name, building 
from instructor, department
where instructor.dept_name = department.dept_name;
```

通常来说，一个SQL查询的含义可以理解如下：

1. 为from子句中列出的关系产生笛卡尔积
2. 在步骤1的结果上应用where子句中指定的谓词
3. 对于步骤2结果中的每个元组，输出select子句中指定的属性

## 第4节 SQL的数据查询能力

### 4.1 聚集函数

聚集函数是以值的一个集合（集或多重集）为输入、返回单个值的函数。SQL提供了五个固有聚集函数：

- 平均值：avg
- 最小值：min
- 最大值：max
- 总和：sum
- 计数：count

####  SQL查询能力很强：

1. 实现了基本代数运算
2. 灵活的表间连接方式
3. 实现了代数运算复合(下面的嵌套子查询)
4. 灵活的where条件
5. 聚集函数等常用函数
6. 嵌入式和动态SQL

#### 案例

- 仅计算一个系的平均工资

```mysql
select avg (salary)
from instructor
where dept_name= 'Comp. Sci.';
```

- 计数前先去除重复元组

```mysql
select count (distinct ID)
from teaches
where semester ='Spring' and year=2010;
```

- *代表选择所有属性

```mysql
select count (*)
from course;
```

- 第1个为平均工资显示部门名，第2个用于指定计算范围(分组)

```mysql
select dept_name, avg (salary) as avg_salary
from instructor
group by dept_name;
```

- 限定输出哪些平均工资(结果筛选)

```mysql
select dept_name, avg (salary)
from instructor
group by dept_name
having avg (salary) > 42000;
```

### 4.2 嵌套子查询

#### 什么是SQL嵌套子查询？

- 子查询是嵌套在另一个查询中的select-from-where表达式。子查询嵌套在where子句中，通常用于对集合的成员资格、集合的比较以及集合的基数进行检查。

#### 4.2.1 集合成员资格

SQL允许测试元组在关系中的成员资格。连接词in测试元组是否是集合中的成员，集合是由select子句产生的一组值构成的。连接词not in则测试元组是否不是集合中的成员。

找出在2009年秋季，但不在2010年春季同时开课的所有课程

```mysql
select distinct course_id
from section
where semester = ’Fall’ and year= 2009 and course_id  not in (select course_id    
                    from section
                    where semester = ’Spring’ and year= 2010);

```

#### 4.2.2 集合的比较

查出这些老师的姓名，他的工资要比Biology系某教师工资高

```mysql
select name
from instructor
where salary > some (select salary
                  from instructor
                  where dept_name ='Biology');
```

找出平均工资最高的系

```mysql
select dept_name
from instructor
group by dept_name
having avg(salary) >= all(select avg(salary)
                          								from instuctor
                          								group by depat_name);
```

#### 4.2.3 空关系测试

找出在2009年秋季和2010年春季同时开课的所有课程

```mysql
select course_id
from section as S
where semester='Fall' and year=2009 and
exist (select *
                from section as T
                where semester='Sring' amd year=2010 and
								S.course_id=T.course_id);	
```

#### 4.2.4 属性的别名

```mysql
select dept_name, avg_salary
from (select dept_name, avg(salary) as avg_slary
           from instructor
           group by depat_name)   as B
where avg_salary >42000;
```

#### 4.2.5 表的别名

```mysql
select dept_name,
           (select count(*)
             from instructor
             where department.dept_name=instructor.dept_name)
             as num_instructors
from department;
```

### 4.3 数据库的修改

#### 4.3.1 删除数据（可利用嵌套子句）

```mysql
delete from instructor
where dept_name= ’Finance’;

delete from instructor
where dept_name in (select dept_name 
                    from department 
                    where building = ’Watson’);
```

#### 4.3.2 插入数据

- ```mysql
  insert into course (course_id, title, dept_name, credits)
  values (’CS-437’, ’Database Systems’, ’Comp. Sci.’, 4);
  ```

- ```mysql
  insert into student
  values (’3003’, ’Green’, ’Finance’, null);
  ```

- ```mysql
  insert into instuctor 
  								select ID, name, dept_name, 18000 
  								from student
  								where dept_name = ‘Music’ and tot_cred > 144;
  ```

#### 4.3.3 更新数据

```mysql
update instructor
set salary = salary * 1.03
where salary > 100000;
update instructor
set salary = case
							when salary <= 100000 then salary * 1.05
              else salary * 1.03
              end;
```

## 第5节 SQL支持的表间连接方式

![图片6](https://tva1.sinaimg.cn/large/007S8ZIlly1gfixc6qheuj31180ni0z5.jpg)

