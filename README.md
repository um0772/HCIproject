# LSTM, CNN을 통한 텍스트와 이미지 분류

CNN 텍스트 분류
- Kaggle에서 사람들의 감정을 텍스트로 표현한 데이터셋을 선택하여 가져왔다.
- 코랩에서 진행하였고 파이토치를 사용했다.
- 먼저, 사용하려는 데이터셋을 불러오고, 불러온 데이터셋을 학습 및 테스트로 분할했다.
- 이 다음은 CNN으로 텍스트를 분류하는 과정이다.

1. 전처리 - 텍스트를 로드한 후, 정리, 토큰화, 패딩 학습 및 테스트를 위한 전처리 파이프라인 생성
2. 텍스트 정리는 특수 문자와 숫자를 제거하고 소문자로 변환한다.
3. 단어 토큰화 - 데이터 셋에서 가장 자주 사용되는 단어 "x"로 사전을 생성해야 함
4. 단어 사전을 작성하는데 인덱스 1로 시작(패딩을 적용하기 위한 인덱스 0을 예약하기 때문)
5. 각 단어 토큰을 숫자 형식으로 변환해야 하므로 앞서 생성한 사전을 사용하여 각 단어를 인덱스 기반 표현으로 변환

- 패딩 : 단어 수가 동일한 것이 중요(패딩을 쓰는 이유)
