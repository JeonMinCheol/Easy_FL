# Easy_FL 설치 및 실행 가이드

이 문서는 수업 시간에 사용할 `Easy_FL` 프로젝트의 환경 설정 및 실행 방법을 안내합니다.

## 1. 가상환경 생성 (권장)

파이썬 패키지 충돌을 방지하기 위해 가상환경을 생성하여 진행하는 것을 권장합니다.

**conda를 사용하는 경우:**

conda create -n easyfl python=3.8

conda activate easyfl

**기본 venv를 사용하는 경우:**
### Windows

python -m venv venv

.\venv\Scripts\activate

### Mac/Linux

python3 -m venv venv

source venv/bin/activate

## 2. 라이브러리 설치
### 이 프로젝트는 PyTorch와 설정 파일 로드를 위한 PyYAML가 필요합니다.

PyTorch 설치

https://pytorch.org/get-started/locally/ 이동해서 자신에게 맞는 버전 설치.

pyyaml 설치

pip install pyyaml

## 3. 설정 파일 (config.yml) 작성

프로젝트 최상위 폴더(루트 디렉토리)에 config.yml 파일을 생성하고 아래 내용을 붙여넣으세요. (예시입니다.)

    Server:
  
      ip: 127.0.0.1        # 서버 IP 주소 (로컬 테스트 시 127.0.0.1)
    
      port: 10000          # 통신 포트
    
      join_ratio: 0.5      # 라운드 참여 비율
    
      rounds: 50           # 전체 라운드 수
    
      timeout: 200         # 타임아웃 시간 (초)
    
      number_of_clients: 3 # 전체 클라이언트 수
  
    Client:
  
      batch_size: 32       # 배치 크기
    
      learning_rate: 0.001 # 학습률
    
      local_epochs: 5      # 로컬 학습 에폭 수
    
      timeout: 200         # 타임아웃 시간
  
    Dataset:

      name: FMNIST         # 데이터셋 이름
  
      num_classes: 10      # 클래스 개수

    Model:

      type: MLP
  
      in_features: 3072
  
      hidden_dim: 10

## 4. 실행 방법
서버와 클라이언트는 각각 다른 터미널 창을 열어서 실행해야 합니다.

1단계: 서버 실행

첫 번째 터미널에서 서버를 실행합니다.

python servers/server.py


2단계: 클라이언트 실행

새로운 터미널 창을 열고 클라이언트를 실행합니다. 

설정 파일의 number_of_clients만큼의 클라이언트가 접속해야 학습이 시작됩니다.

python clients/client.py (number_of_clients만큼 실행)


모든 클라이언트가 연결되면 서버에서 자동으로 연합 학습(Federated Learning) 라운드가 시작됩니다.

