Radar Gesture Recognition

基于多视图雷达特征融合的手势识别项目
Multi-view Radar Gesture Recognition with MVCA and Vertical Discrimination

本项目使用 PyTorch 构建雷达手势分类系统，联合处理 RT、DT、RDT、ERT、ART 五种雷达特征，支持多视图特征编码、注意力融合、训练与测试、可选垂直方向判别器（Vertical Discriminator, VD）、消融实验及特征可视化。项目提供命令行训练/评估入口和基于 Qt 的桌面界面。

说明：本文根据仓库内实际源代码与 YAML 配置整理。当前代码包包含训练/分析日志，但不包含 dataset/ 原始特征数据、save_model/ 训练权重或完整的依赖锁定文件。首次运行前须自行准备数据并配置环境。本文未对运行结果进行重新训练验证。

目录

项目功能

项目结构

环境配置

数据集准备

快速开始

配置文件说明

单独评估已训练模型

消融实验和可视化

运行结果

常见问题

项目功能

五视图特征建模：融合 RT、DT、RDT、ERT、ART 雷达特征；使用二维特征编码器和三维/时序特征编码器。

多视图融合：支持 concatenate 和 attention 两类融合方式；注意力路径实现了增强型 CSAM/MVCA 及其组件消融变体。

垂直动作辅助识别：可选 VD 分支，针对 Down / Up 增加方向判别与辅助损失。

训练策略：包含数据增强、类别加权、可选类别均衡采样、域对抗学习、辅助损失及最佳检查点保存。

评估与分析：支持测试集 Top-1 准确率、混淆矩阵、逐类准确率、三随机种子实验，以及视图/模块消融和 t-SNE 可视化。

两种使用方式：命令行运行训练及测试，或通过 Qt 桌面界面交互操作。

手势类别

代码中的 class_mapping 定义了 8 个类别：

编号

标签

含义

0

Back

向后

1

Dblclick

双击

2

Down

向下

3

Front

向前

4

Left

向左

5

Right

向右

6

Up

向上

7

Unknown

未知类别

项目结构

.
├── bin/                         # 训练、数据加载、测试与实验脚本
│   ├── dataset.py               # MultiViewDataset：读取五视图 .npy 特征
│   ├── train.py                 # 模型训练和测试核心逻辑
│   ├── train_utils.py           # 单轮训练及增强/损失相关逻辑
│   ├── validate_utils.py        # 验证与评估
│   ├── ride_augmentation.py     # 数据增强
│   ├── test_specific_checkpoint.py
│   ├── profile_model.py        # 模型复杂度与推理耗时分析
│   ├── run_mvca_ablation.py     # MVCA 组件消融实验
│   ├── run_vd_ablation_3seeds.py
│   ├── run_view_ablation_3seeds.py
│   ├── export_tsne_vertical_features.py
│   └── export_view_weights_3seeds.py
├── models/                      # 网络模型、编码器、融合器和注意力模块
│   ├── model.py                 # MultiViewFeatureFusion
│   ├── encoder.py
│   ├── decoder.py
│   ├── enhanced_csam.py
│   ├── csam.py
│   └── methods/
├── examples/                    # YAML 训练配置与消融配置
│   ├── conf_mvca_full.yaml
│   ├── confB_enhanced_csam1_upboost.yaml
│   ├── conf_mvca_channel_only.yaml
│   └── conf_mvca_crossview_only.yaml
├── gui/
│   ├── main_not_gui.py          # 推荐的命令行训练 + 自动测试入口
│   ├── main.py                  # Qt 桌面界面入口
│   └── interface.py             # 界面组件
├── utils/                       # 检查点、指标与通用工具
├── logs/                        # 包中已有的历史实验日志/CSV/图表
├── dataset/                     # [需自行准备] 输入特征数据
├── save_model/                  # [运行后生成] 模型权重与结果
└── README.md

环境配置

建议使用 Python 3.10 或更新版本，并在项目根目录运行以下命令。代码未提供固定版本的 requirements.txt；以下是依据实际 import 语句整理的依赖清单，不是经过逐版本验证的锁定环境。

# 1. 创建并激活独立环境（可选）
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell（请单独执行这一行）
# .venv\Scripts\Activate.ps1

# 2. 安装 PyTorch（CPU / CUDA 版本请按设备选择）
pip install torch torchvision

# 3. 安装项目其余依赖
pip install numpy PyYAML matplotlib pandas scikit-learn thop pyqtgraph PySide6

如果需要 GPU 训练，请在 PyTorch 官方安装页面 选择与你的系统及 CUDA 环境匹配的安装命令。bin/train.py 会优先使用可用的 CUDA GPU，否则尝试使用 CPU；CPU 训练可能较慢。

仅使用命令行训练时，可以不安装 GUI 专用的 pyqtgraph 和 PySide6。运行 gui/main.py 则需要桌面图形环境和相关依赖。

数据集准备

1. 准备目录

配置文件默认使用 dataset/dataV/person_1 至 dataset/dataV/person_10，部分实验还使用 dataset/dataS。示意目录如下：

dataset/
├── dataV/
│   ├── person_1/
│   │   ├── 00_Back/
│   │   │   ├── ARTsample_0000001.npy
│   │   │   ├── DTsample_0000001.npy
│   │   │   ├── ERTsample_0000001.npy
│   │   │   ├── RDTsample_0000001.npy
│   │   │   └── RTsample_0000001.npy
│   │   ├── 01_Dblclick/
│   │   └── ...
│   ├── person_2/
│   └── ...
└── dataS/                      # 可选：跨场景测试/部分消融实验

上面的文件名与类别目录名是格式示例，不是项目随附的真实数据。需要将你已有的数据转换成与 bin/dataset.py 兼容的格式。

2. 特征格式

MultiViewDataset 使用 numpy.load() 读取 .npy 文件。默认配置中的单样本输入形状为：

特征

.npy 原始数组形状

读入后张量形状（不含 batch）

RT

(48, 64)

(1, 48, 64)

DT

(12, 64)

(1, 12, 64)

RDT

(12, 64, 64)

(12, 1, 64, 64)

ERT

(12, 91, 64)

(12, 1, 91, 64)

ART

(12, 91, 64)

(12, 1, 91, 64)

注意实际数据加载器的约束：

同一样本必须存在 ART、DT、ERT、RDT、RT 五个 .npy 文件；即使模型只选择部分视图，当前数据加载器仍会读取全部五种特征。

文件名使用视图前缀加相同的末尾 18 个字符，例如 ARTsample_0000001.npy 与 RTsample_0000001.npy。加载器通过 feature_name + os.path.basename(file_path)[-18:] 找到同一样本的其他特征。

bin/dataset.py 对类别文件夹执行 sorted(os.listdir(...))，并将排序位置直接作为标签索引。务必确保文件夹的字典序与 class_mapping 的编号一致。建议使用 00_Back、01_Dblclick、…、06_Up、07_Unknown 等带前缀的名称；仅使用英文类名时，Unknown 和 Up 的排序可能与映射不符。

同一类别下需按每个样本五个文件的约定组织数据，避免将无关文件放入类别目录；加载器会基于文件数除以 5 计算样本数。

3. 默认的数据划分

在 examples/confB_enhanced_csam1_upboost.yaml 中：

用途

默认目录

训练

person_1～person_4

验证

person_5

测试

person_6～person_10

如果你的数据存放位置不同，请修改 YAML 中的 train_and_vali_data_dir、val_data_dir、test_data_dir。以上相对路径以运行命令时的项目根目录为基准。

快速开始

方式一：命令行训练并自动测试（推荐）

在项目根目录执行：

python gui/main_not_gui.py --config examples/confB_enhanced_csam1_upboost.yaml

该配置使用五种输入视图、attention 融合及 csam_enhanced（完整 MVCA 的兼容名称），默认训练 50 epochs，随后在配置指定的测试数据上评估。

如果只想检查训练链路，可复制配置文件，把副本中的 epochs 改为 1，再使用副本运行。仓库中的 examples/conf_mvca_full.yaml 本身设为 epochs: 1，也可以用于短流程检查：

python gui/main_not_gui.py --config examples/conf_mvca_full.yaml

注意：这里的 1 epoch 是流程检查配置，不代表完整训练结果。首次运行还必须先准备好 YAML 引用的数据路径。

方式二：仅训练

python bin/train.py --config_file examples/confB_enhanced_csam1_upboost.yaml --device 0

该入口直接调用训练流程，不会像 gui/main_not_gui.py 那样接着执行整套自动测试与多次实验汇总。GPU 设备设置请结合运行环境确认。

方式三：启动桌面 GUI

python gui/main.py --config examples/confB_enhanced_csam1_upboost.yaml

该入口通过 Qt 显示数据/特征可视化界面以及训练、测试相关控件，需要图形桌面环境。建议显式传入完整配置文件：gui/main.py 的默认值指向 examples/conf1.yaml，该文件的设置与当前较新的训练入口不完全一致。

继续训练已有模型

gui/main_not_gui.py 支持从已存在的检查点加载权重：

python gui/main_not_gui.py \
  --config examples/confB_enhanced_csam1_upboost.yaml \
  --train-model /path/to/model_best.pth.tar

/path/to/model_best.pth.tar 是占位路径；请替换为实际文件，且确保模型结构与该配置兼容。

配置文件说明

训练主要由 examples/*.yaml 控制。常用字段如下：

参数

作用

train_and_vali_data_dir

训练数据目录列表

val_data_dir

独立验证数据目录

test_data_dir

测试数据目录

epochs、batch_size、lr

训练轮数、批量大小、学习率

seed、deterministic

随机种子和确定性选项

selected_features

实际输入模型的视图集合

input_feature_shapes

每一种视图的张量形状

fusion_mode

如 attention、concatenate

method

具体融合方法，如 csam_enhanced、mvca_full

is_domain_loss

域对抗损失相关设置

is_weights_loss、mv_feature_weights

视图权重先验相关设置

use_vertical_discriminator

启用/禁用 VD 垂直判别器

up_aux_weight、down_aux_weight

Up/Down 辅助损失权重

model_path

模型检查点保存根目录

常见配置文件：

examples/confB_enhanced_csam1_upboost.yaml：增强 CSAM/MVCA + VD 训练配置（50 epochs）。

examples/conf_mvca_full.yaml：完整 MVCA 配置（当前文件是 1 epoch）。

examples/conf_mvca_channel_only.yaml：仅通道/空间注意力分支的消融配置。

examples/conf_mvca_crossview_only.yaml：仅跨视图注意力分支的消融配置。

examples/conf_transfer_continue_template.yaml：继续训练相关配置模板。

VD 使用约束：当 use_vertical_discriminator: true 时，selected_features 必须同时包含 RDT 和 ERT，否则模型初始化会报错。

单独评估已训练模型

指定一次训练的输出目录、检查点和测试数据目录：

python bin/test_specific_checkpoint.py \
  --run-dir /path/to/a/training-run \
  --checkpoint /path/to/a/training-run/model_best.pth.tar \
  --test-dirs dataset/dataV/person_6 dataset/dataV/person_7 \
  --tag manual_test

--run-dir 指向包含 output/args.yaml 的训练输出目录。

--checkpoint 指向实际生成的 model_best.pth.tar 或 checkpoint.pth.tar。

--test-dirs 可传入一个或多个测试数据文件夹。

评估结果会写到该 run 的 output/，包括混淆矩阵、逐类准确率及错误样本列表。

消融实验和可视化

这些命令应在项目根目录执行，且需要准备好配置引用的训练/测试数据。下面的消融脚本通常会启动多轮训练，耗时和 GPU 需求高于单次测试。

1. MVCA 模块消融

python bin/run_mvca_ablation.py --device 0 --skip-datas

比较通道/空间分支与跨视图注意力分支，使用固定随机种子 123、314、456。--skip-datas 只跳过额外的 dataset/dataS 评估，并不会跳过配置指定的常规测试集。

2. 垂直判别器（VD）消融

python bin/run_vd_ablation_3seeds.py --device 0 --skip-datas

在其余主要训练设置不变的前提下禁用 VD，统计多随机种子下的整体性能、Up/Down 准确率及相关混淆情况。

3. 五视图贡献消融

python bin/run_view_ablation_3seeds.py --device 0 --skip-datas

比较完整五视图模型与分别去掉 RT、DT、RDT、ERT、ART 的配置。该脚本的目标是继续运行五个去除单视图的变体，并假定完整模型的对照实验已单独完成；实验中所有变体统一关闭 VD，以减少额外变量干扰。

4. 模型复杂度分析

python bin/profile_model.py --config examples/conf_mvca_full.yaml --device cpu

用于估算模型参数量、GFLOPs 和推理时延；测量结果受硬件及软件环境影响。

5. 可视化工具

仓库包含 bin/export_tsne_vertical_features.py、bin/export_view_weights_3seeds.py 和 bin/export_view_weights_models_3seeds.py，用于提取融合特征、执行 t-SNE 分析以及统计/绘制各视图权重。这类脚本通常依赖已有训练目录与权重，运行前请先使用 --help 查看参数，并检查脚本中的默认实验路径。

已有日志中包含 t-SNE 输出示例（如果同时上传 logs/ 目录，GitHub 可直接显示）：



运行结果

训练产物默认写入 model_path 指定的目录（常见为 ./save_model/），按所选视图、融合方式和方法组织子目录。具体时间戳等路径以运行输出为准。

save_model/
└── ...某次训练目录.../
    ├── checkpoint.pth.tar           # 最新保存的检查点
    ├── model_best.pth.tar           # 选出的最佳检查点（若产生）
    └── output/
        ├── args.yaml               # 本次运行参数
        ├── log.txt
        ├── confusion_matrix.txt    # 测试/评估输出（按入口而异）
        └── ...

gui/main_not_gui.py 的多次实验摘要保存在 logs/multi_run_summary/ 下，消融实验的 CSV 与日志会写入 logs/mvca_ablation_3seeds/、logs/vd_ablation_3seeds/、logs/view_ablation_3seeds/ 等目录。压缩包已附带部分历史实验记录，但不应将历史记录直接视为重新运行后必然得到的精度。

常见问题

Q1：提示 FileNotFoundError: dataset/...？
数据集不在当前代码包内。确认 dataset/ 已放在项目根目录，或者修改 YAML 中的数据路径；同时从项目根目录执行命令。

Q2：提示找不到 .npy 特征文件？
检查五种视图文件是否成套存在，且同一样本的文件名末尾 18 个字符一致；查看 bin/dataset.py 中 load_and_expand() 的文件路径拼接规则。

Q3：准确率或类别标签明显不对？
先核对类别子目录的字典序是否与 class_mapping 一致。该项目的数据加载器使用目录顺序作为分类标签，不会自动依据文件夹英文名查表。

Q4：怎么使用 GPU？
确认安装了 CUDA 兼容版 PyTorch，并运行 python -c "import torch; print(torch.cuda.is_available())"。默认训练核心在检测到 CUDA 后使用 cuda:0；选择其他物理 GPU 时，可设置 CUDA_VISIBLE_DEVICES。

Q5：GitHub 上传后为什么看不到 README？
请把本文件以**README.md** 为文件名放在 GitHub 仓库根目录，并与 bin/、models/、examples/ 等文件夹处于同一层级。仅上传 ZIP 压缩包，GitHub 不会自动解压并读取里面的 README。

Q6：可以公开上传数据和模型吗？
公开仓库前请确认你有权分享数据、模型权重和实验日志。一般建议不要提交 .venv/、__pycache__/、大型数据集和训练中间权重。当前代码包未发现项目根目录的 LICENSE，因此不要擅自标注 MIT 等许可证；如准备开源，请由项目所有者确定并添加适合的许可文件。

使用说明：本 README 与当前上传压缩包中的文件结构对应；所有 /path/to/... 均为需要替换的占位路径。实际运行成功与否取决于数据格式、环境依赖、配置及权重文件是否完整。
