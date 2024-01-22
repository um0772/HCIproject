# LSTM, CNN을 통한 텍스트와 이미지 분류

CNN 텍스트 분류
- Kaggle에서 사람들의 감정을 텍스트로 표현한 데이터셋을 선택하여 가져왔다.
- 코랩에서 진행하였고 파이토치를 사용했다.
- 먼저, 사용하려는 데이터셋을 불러오고, 불러온 데이터셋을 학습 및 테스트로 분할했다.
- 이 다음은 CNN으로 텍스트를 분류하는 과정이다.

전처리 - 텍스트를 로드한 후, 정리, 토큰화, 패딩 학습 및 테스트를 위한 전처리 파이프라인 생성

1. 텍스트 정리는 특수 문자와 숫자를 제거하고 소문자로 변환한다.
2. 단어 토큰화 - 데이터 셋에서 가장 자주 사용되는 단어 "x"로 사전을 생성해야 함
3. 단어 사전을 작성하는데 인덱스 1로 시작(패딩을 적용하기 위한 인덱스 0을 예약하기 때문)
4. 각 단어 토큰을 숫자 형식으로 변환해야 하므로 앞서 생성한 사전을 사용하여 각 단어를 인덱스 기반 표현으로 변환
5. 각 단어의 길이를 표준화하기 위해 패딩을 구현해야 함
6. 패딩 - 단어 수가 동일해야 하고, 사전을 구축할 때 예약한 인덱스인 0을 사용한다.
7. 전처리 파이프라인의 마지막으로 훈련과 테스트로 나눈다.

TextCNN 모델 클래스 정의
1. 임베딩 레이어로 단어를 임베딩 벡터로 변환
2. 지정된 크기의 다양한 커널 크기를 가진 합성곱 레이어를 생성
3. 여러 개의 합성곱 레이어를 관리하기 위한 리스트 생성
4. 과적합을 방지하기 위해 합성곱 레이어의 출력에 드롭아웃을 적용
5. 최종 출력을 클래스 수에 맞게 완전 연결 레이어 변환
6. 모델을 인스턴스화 할 때 필요한 매개변수들을 전달하여 모델을 초기화(단어장 크기, 단어 임베딩 차원, 클래스의 개수, 합성곱 레이어에 사용될 커널의 크기, 드롭아웃)
-> 최종적인 모델 생성

train 함수: 모델을 훈련 모드로 설정, 훈련 중에는 모델이 가중치를 업데이트하고 그래디언트를 계산한다. 배치마다 입력 데이터와 레이블을 모델에 주입하여 예측을 수행하고 손실을 계산한다. 손실을 역전파하여 모델의 가중치를 업데이트하고, 에폭 당 평균 훈련 손실을 반환한다.

evaluate 함수: 모델을 평가 모드로 설정하고, 검증 또는 테스트 데이터로 모델의 성능을 평가하는 역할을 한다. 배치마다 입력 데이터와 레이블을 모델에 주입하여 예측을 수행하고 손실을 계산한다. 손실을 이용하여 모델의 성능을 측정하며, 에폭 당 평균 테스트 손실을 반환한다.
마지막으로, 훈련과 테스트 손실을 에폭 별로 기록하고, 이를 시각화하기 위해 그래프를 출력했다. 

결과


![image](https://github.com/um0772/HCIproject/assets/58669248/6948b6c2-514c-4f86-bffa-267bf04f876f)

- Accuracy(정확도): 0.8653
- Precision(정밀도): 0.8656
- Recall(재현율): 0.8653
- F1 Score(F1 점수): 0.8626

추가로 예측하고자 하는 텍스트를 입력했을 때 주어진 텍스트에 대한 감정을 예측하는 코드 작성

- “I am feeling very happy today!” 이 텍스트를 입력했을 때
- Predicted label: happy 라는 결과가 나오는 것을 확인할 수 있음
- 위의 과정을 통해 입력한 텍스트를 분류하여 어떠한 감정을 가진 문장인지 파악한 후 결과 출력

