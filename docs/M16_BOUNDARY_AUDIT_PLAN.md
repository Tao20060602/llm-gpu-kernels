# M16 冻结后历史长度上限审计

原44项新输入中短query规则没有观测到回归，但它们最大history为2625，
冻结适用域上限为4096。本审计在查看这些新输入后提出，属于追加验证，
不伪称原始预注册holdout；配置、阈值和域均保持不变。

固定seed14307，T32/48/64/96/127/128 × selected3073/3519/3520/3521，共24；
另T64/selected8193域外1项，共25。sink64/recent512/currentT，history为
3649/4095/4096/4097；4097和8769应回退FA2，≤4096依旧用原冻结规则。
使用M16相同四arm完整API/Graph协议：4轮正/逆，8样本2调用、Graph8forward；
每arm计时前后独立chunk64 FP32、原0.008/finite、输入hash。验证原31项冻结
source+training/policy provenance及本审计2项source hash运行前后。

如上限处回归，不修改旧policy、不事后缩域，规则保持实验用途；该追加审计不
证明连续形状空间或真实模型质量。全部raw数据保留。
