## 第1节 SQL查询语言概览

- **数据定义语言（Data-Definition Language，DDL）**：SQL DDL提供定义关系模式、删除关系以及修改关系模式的命令。
- **数据操纵语言（Data-Manipulation Language，DML）：**SQL DML提供从数据库中查询信息，以及在数据库中插入元组、删除元组、修改元组的能力。
- **完整性**： SQL DDL包括定义完整性约束的命令，保存在数据库中的数据必须满足所定义的完整性约束。破坏完整性约束的更新是不被允许的。
- **视图定义：**SQL DDL包括定义视图的命令。
- **事务控制：**SQL包括定义事务的开始和结束的命令。
- 嵌入式SQL和动态SQL

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

### 第2节 SQL数据定义

数据库中的关系集合必须由数据定义语言（DDL）指定给系统。SQL的DDL不仅能够定义一组关系，还能够定义每个关系的信息。

#### 3.2.1 基本类型

- char(n)：固定长度的字符串，用户指定长度n
- varchar(n)：可变长度的字符串，用户指定最大长度n
- int(n)：整数类型（和机器相关的相关的整数类型的子集）
- numeric(p, d)：定点数，精度由用户指定。这个数有p位数字（加上一个符号位），其中d位数字在小数点右边。
- real，double precision：浮点数与双精度浮点数，精度与机器相关。
- float(n)：精度至少位n位的浮点数

#### 3.2.2 基本模式定义

```sql
create tabel department
( dept_name varchar(20),
  building varchar(15),
  budget numeric(12,2),
  primary key (dept_name));
```

