# BiSeNet on Point Cloud data
Point Cloud를 구면좌표계로 투영시키켜서 2D 이미지를 만들고 이것을 기존 이미지 시맨틱 세그멘테이션 알고리즘인 BiSeNet을 활용하여 세그멘테이션하는 모델입니다.

# How to train
본 모델은 Semantic KITTI 데이터셋을 활용하여 학습이 진행되었습니다. 학습에는 config와 실제 data가 필요하며 config는 ./dataset/semanticKITTI/config에 data_cfg.yaml, semantic-kitti-all.yaml, semantic-kitti.yaml 파일로 저장되어있습니다. 학습을 위한 설정들은 data_cfg.yaml 파일에 기록되어 있어 수정이 필요한 경우 yaml 파일을 수정하면 됩니다.
실제 데이터는 ./dataset/semanticKITTI/Sequences에 시퀀스별로 인덱스가 매겨져 저장되어 있으며 각 시퀀스 내에는 velodyne과 labels 폴더가 있어서 velodyne 폴더 내부에는 point cloud 데이터가 bin 파일로, labels 폴더 내부에는 gt가 label파일로 저장되어있어야 합니다.
데이터가 제대로 위치해 있다면 기본적인 설정으로 학습을 진행시키기 위해서는 다음과 같이 콘솔에 입력하면 됩니다:
'''console
cd tools
python train_pc.py
'''
학습결과는 root 디렉토리에 res 폴더가 생기며 그곳에 2 epoch 마다 저장됩니다.

Semantic KITTI는 point cloud 데이터셋이지만, 실제 모델은 구면좌표계에 투영된 이미지를 input으로 받기 때문에 모델에 입력되기전에 변환이 되어야합니다. 해당 과정은 ./datasets/semanticKITTI 에 위치한 parser.py 및 laserscan.py, laserscanvis.py에 의해 진행됩니다. parser는 ./tools/train_pc.py 파일로 import 되어 호출되어 좌표변환된 데이터를 공급하는 dataloader를 반환합니다.

Fine Tuning을 위해서 학습은 저장된 checkpoint에서 부터 시작할 수도 있습니다. 이를 위해선 train_pc.py를 실행시킬때 --pth_dir 와 --start_epoch 매개변수를 추가해주면 됩니다. --pth_dir의 경우 저장된 checkpoint파일(*.pth)의 주소를 입력하고 --start_epoch은 불러온 checkpoint파일에서 학습을 재개할 때 시작할 epoch 입니다. 가령, 70번째 epoch까지 학습이 진행된 checkpoint로 부터 학습을 재개할 때, --start_epoch 매개변수로 71을 입력해야합니다. 예시는 다음과 같습니다:
'''console
cd tools
python train_pc.py --pth_dir (path_to_pth_file) --start_epoch (starting_epoch)
'''

# How to evaluate
모델의 성능 평가는 변환된 구면좌표계의 차원에서의 라벨과 비교하여 진행됩니다. 