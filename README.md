# 실행 방법
1. yml 파일 설정
2. prepare_kdd99_from_pickle.py 실행 (서버, 클라이언트 둘 다)
3. 서버는 server.py 실행
4. 클라이언트는 client.py 실행


## 1) 설정 파일 확인

기본 `config.yml`은 다음 방향으로 맞춰져 있습니다.

- `Server.number_of_clients`: 전처리 때 만든 client 수와 동일해야 함
- `Dataset.name: KDD99`
- `Dataset.threshold_method: std`
- `Dataset.threshold_std_factor: 3.0`

원하면 다음도 바꿀 수 있습니다.

```yaml
Client:
  batch_size: 256
  learning_rate: 0.0001
  local_epochs: 3

Dataset:
  threshold_method: quantile
  threshold_quantile: 0.99

Model:
  latent_dim: 4
  dropout: 0.1
  hidden_dims: [96, 64, 48, 16]
```


## 1) KDD99 데이터 전처리 및 client shard 생성

### `preprocessed_data_full.pkl`이 있어야 합니다.
예:
```bash
python prepare_kdd99_from_pickle.py --pickle_path /path/to/preprocessed_data_full.pkl --num_clients 3 --output_dir dataset/KDD99 --train_anomaly_fraction 0.0
```

num_clients 인자에 yml 파일과 동일한 클라이언트 수를 입력해주세요.

### 중요!
서버 또한 클라이언트와 같은 수의 데이터 샤드를 미리 생성해야합니다.
(가중치 전송에 사용됨.)

## 3) 실행 방법

서버 컴퓨터에서 실행:

```bash
python servers/server.py
```

다른 터미널 혹은 컴퓨터들에서 client 실행 (`number_of_clients` 만큼):

```bash
python clients/client.py
```

## 4. 모델 저장 및 로컬 추론

학습이 진행되면 다음 파일들이 자동으로 저장됩니다.

- 서버 글로벌 모델: `artifacts/server/global_model_latest.pt`
- 클라이언트 로컬 체크포인트: `artifacts/clients/client_<id>_latest.pt`

클라이언트 PC에서 저장된 체크포인트로 테스트 shard를 바로 평가하려면:

```bash
python clients/infer_saved_model.py   --checkpoint_path artifacts/clients/client_0_latest.pt   --client_id 0   --split test
```

특정 npz 파일을 직접 평가하려면:

```bash
python clients/infer_saved_model.py   --checkpoint_path artifacts/clients/client_0_latest.pt   --npz_path dataset/KDD99/test/0.npz
```

출력은 기본적으로 `threshold`, `loss`, `acc`, `predicted_anomalies`를 보여줍니다.
