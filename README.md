BiSeNet on Point Cloud data
===========================
Point Cloud를 구면좌표계로 투영시키켜서 2D 이미지를 만들고 이것을 기존 이미지 시맨틱 세그멘테이션 알고리즘인 BiSeNet을 활용하여 세그멘테이션하는 모델입니다.

#### 참고사항
이후에 서술할 설명들에서 상대 위치들과 console 명령어들은 모두 root 디렉토리를 기준으로 기술되어 있습니다.

## About Semantic KITTI
본 모델은 초반에는 일반적인 KITTI 데이터셋을 활용하여 학습 및 성능 평가가 진행되었으며 Car, Cyclist, Pedestrian, unlabeled 4개의 카테고리에 대해 예측을 진행했습니다. ~~그러나, 이 후 더 자세한 라벨링인 된 Semantic KITTI 데이터셋을 활용하여 학습이 진행되었습니다.~~ (2021/08/25 수정) 학습을 다시 진행해야 합니다. Semnatic KITTI는 unlabeled를 포함하여 총 20가지 클래스로 라벨링이 되어 있기 때문에 *./configs/bisenetonpc2.py* 파일내에 저장된 모델 설정중 num_cls 속성을 20으로 설정하면 됩니다.

## References
본 모델은 SqueezeSeg에서 영감을 받아 BiSeNet v2 모델을 적용시켜본 프로젝트입니다. 추가적인 이론적 배경이 필요하시면 해당 논문들을 참고해주세요.

SqueezeSeg: [[paper]][ssg_paper] [[git]][ssg_git]

SqueezeSeg v2: [[paper]][ssg2_paper] [[git]][ssg2_git]

SqueezeSeg v3: [[paper]][ssg3_paper] [[git]][ssg3_git]

BiSeNet v2: [[paper]][bise_paper] [[git]][bise_git]

## How to train
 학습에는 config와 실제 data가 필요하며 config는 *./dataset/semanticKITTI/config*에 *data_cfg.yaml, semantic-kitti-all.yaml, semantic-kitti.yaml* 파일로 저장되어있습니다. 학습을 위한 설정들은 *data_cfg.yaml* 파일에 기록되어 있어 수정이 필요한 경우 yaml 파일을 수정하면 됩니다.
실제 데이터는 *./dataset/semanticKITTI/Sequences*에 시퀀스별로 인덱스가 매겨져 저장되어 있으며 각 시퀀스 내에는 *velodyne*과 *labels* 폴더가 있어서 *velodyne* 폴더 내부에는 point cloud 데이터가 bin 파일로, *labels* 폴더 내부에는 gt가 label파일로 저장되어있어야 합니다.
데이터가 제대로 위치해 있다면 기본적인 설정으로 학습을 진행시키기 위해서는 다음과 같이 콘솔에 입력하면 됩니다:
```console
cd tools
python train_pc.py
```
학습결과는 root 디렉토리에 res 폴더가 생기며 그곳에 2 epoch 마다 저장됩니다.

Semantic KITTI는 point cloud 데이터셋이지만, 실제 모델은 구면좌표계에 투영된 이미지를 input으로 받기 때문에 모델에 입력되기전에 변환이 되어야합니다. 해당 과정은 *./datasets/semanticKITTI*에 위치한 *parser.py* 및 *laserscan.py, laserscanvis.py*에 의해 진행됩니다. parser는 *./tools/train_pc.py* 파일로 import 되어 호출되어 좌표변환된 데이터를 공급하는 dataloader를 반환합니다.

Fine Tuning을 위해서 학습은 저장된 checkpoint에서 부터 시작할 수도 있습니다. 이를 위해선 *train_pc.py*를 실행시킬때 ***--pth_dir*** 와 ***--start_epoch*** 매개변수를 추가해주면 됩니다. ***--pth_dir***의 경우 저장된 checkpoint파일(~.pth)의 주소를 입력하고 ***--start_epoch***은 불러온 checkpoint파일에서 학습을 재개할 때 시작할 epoch 입니다. 가령, 70번째 epoch까지 학습이 진행된 checkpoint로 부터 학습을 재개할 때, ***--start_epoch*** 매개변수로 71을 입력해야합니다. 예시는 다음과 같습니다:
```console
cd tools
python train_pc.py --pth_dir (path_to_pth_file) --start_epoch (starting_epoch)
```

## How to evaluate
모델의 성능 평가는 변환된 구면좌표계의 차원에서의 라벨과 비교하여 진행됩니다. 성능 평가에서는 학습시 config를 사용하므로 별도로 yaml파일을 작성할 필요가 없습니다. 성능 평가는 기본적으로 *./res/model_final.pth*를 가져와서 진행되도록 설정되어 있으나 원하는 checkpoint 파일로 실행시 입력해줄 수 있습니다. 학습시와 동일하게 ***--pth_dir*** 매개변수를 추가하여 원하는 checkpoint 파일의 디렉토리를 입력해주면 됩니다. 성능 평가를 cpu를 사용하여 진행하고 싶을때는 ***--cpu*** 옵션을 사용하면 cpu에서 성능평가가 진행되도록 되어있습니다. 이 밖에도 ***--model*** 옵션과 ***--data_dir*** 옵션이 있어서 모델과 데이터를 설정할 수 있으나, 모델의 경우 bisenetonpc2를 테스트하기 위해 따로 설정할 필요가 없고, 데이터 또한 SemanticKITTI를 사용하지 않는 경우 외에는 설정할 필요가 없습니다. 예시는 다음과 같습니다:
```console
cd tools
python evaluate_pc.py --pth_dir (path_to_pth_file) --cpu
```

## Customize
모델 파일들은 *./lib/models* 폴더 내에 위치해 있으므로 필요하다면 해당 디렉토리내에 새 모델을 만들거나 기존 모델을 수정할 수 있습니다. (2021/08/25 기준) 최신 모델은 디렉토리내에 *bisenet_on_pc_v2.py*에 저장되어 있습니다. 

### Model Factory
*./lib/models* 폴더 내의 *__init__.py* 파일에 *model_factory* dict가 선언되어 있으며 학습 및 성능 평가시 *./configs*에 있는 설정 파일에서 *model_type* 변수를 이용하여 *model_factory*로 부터 필요한 모델을 호출합니다. 따라서, 새로운 모델을 만드면 *model_factory*에 추가하여 config파일에 맞게 설정을 해야합니다.

### bisenet_on_pc.py
point cloud에 적용하기 위해 Detail branch와 Segment branch를 수정한 모델입니다. *BiSeNet_pc* 클래스는 *bisenetv2.py*에 저장된 기본 *BiSeNetV2* 클래스를 상속받아서 수정되었습니다.

### bisenet_on_pc_v2.py
단순히 SqueezeSeg와 BiSeNet v2를 합쳤을 때 만족할 만한 성능이 나오지 않아 여러가지 모듈들을 추가해놓은 모델입니다. 다음과 같은 추가적인 방안들이 구현되어 있습니다:

    1. Context Aggregaton Module (visual attention)
    2. Convolutional Bottleneck Attention Module [[CBAM]][cbam] (visual attention)
    3. Deeper Semantic Branch (deeper model)

*bisenet_on_pc_v2.py* 내의 *BiSeNet_pc2* 클래스는 *bisenetv2.py*에 저장된 기본 *BiSeNetV2* 클래스를 상속받아서 수정되었으며 *bisenet_on_pc.py*에서 *DetailBranch_pc, SegmentBranch_pc*를 import해와서 사용합니다.

## 그 외 기타
그 외 point cloud와 관련하여 추가적인 파일들이 *./tools* 디렉토리 내에 저장되었습니다.
*pc_view.py* 같은 경우 pcl_viewer를 활용하여 point cloud 파일(~.pcd, ~.ply)을 볼 수 있게 해줍니다. *visualizer.py*는 모델로 추출한 구면 좌표계 이미지를 다시 point cloud로 back project할 수 있게 해주는 툴입니다. 각각 모두 테스트해 볼 수 있도록 샘플코드도 추가되어 있습니다.

[ssg_paper]: https://arxiv.org/abs/1710.07368
[ssg_git]: https://github.com/BichenWuUCB/SqueezeSeg
[ssg2_paper]: https://arxiv.org/abs/1809.08495
[ssg2_git]: https://github.com/xuanyuzhou98/SqueezeSegV2
[ssg3_paper]: https://arxiv.org/abs/2004.01803
[ssg3_git]: https://github.com/chenfengxu714/SqueezeSegV3
[bise_paper]: https://arxiv.org/abs/2004.02147
[bise_git]: https://github.com/CoinCheung/BiSeNet
[cbam]: https://github.com/Jongchan/attention-module