import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# spacy 언어 모델 로드
spacy_en = spacy.load('en_core_web_sm')

# Kaggle 데이터셋 다운로드를 위한 라이브러리 설치 및 설정
!pip install Kaggle

from google.colab import files
uploaded = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Kaggle에서 데이터셋 다운로드
!kaggle datasets download -d ishantjuyal/emotions-in-text
!unzip emotions-in-text.zip



!ls

# 데이터셋 불러오기
df = pd.read_csv('Emotion_final.csv')

# 레이블 수치화
label_encoder = LabelEncoder()
df['Emotion'] = label_encoder.fit_transform(df['Emotion'])

# 데이터셋 분할
train_df, test_df = train_test_split(df, test_size=0.2)
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

def label_transform(label):
    return label




import re
# 데이터 셋 클래스 정의
class TextDataset(Dataset):
    def __init__(self, filename, tokenizer, vocab, max_length, label_transform=None):
        self.data = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length
        self.label_transform = label_transform

    def __len__(self):
        return len(self.data)

    def clean_text(text):
      text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
      return text

    def __getitem__(self, idx):
      text = self.data.iloc[idx]['Text'].lower()  # 소문자 변환
      label = self.data.iloc[idx]['Emotion']

      tokenized_text = self.tokenizer(text)
      tokenized_text = [token for token in tokenized_text if token not in spacy.lang.en.stop_words.STOP_WORDS]  # 불용어 제거
      numericalized_text = [self.vocab[token] for token in tokenized_text][:self.max_length]
      padded_text = numericalized_text + [self.vocab['<pad>']] * (self.max_length - len(numericalized_text))

      if self.label_transform:
        label = self.label_transform(label)

      return torch.tensor(padded_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)



!pip install torchtext


from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from collections import Counter
import pandas as pd
# 데이터셋 로드
df = pd.read_csv('Emotion_final.csv')
texts = df['Text']

# 토크나이저 및 단어장 구축
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# 단어의 빈도수 계산
counter = Counter()
for text in df['Text']:
    counter.update(tokenizer(text))

# 단어장 구축
vocab = build_vocab_from_iterator(yield_tokens(texts), specials=['<unk>', '<pad>'])
vocab.set_default_index(vocab['<unk>'])


# 데이터셋 인스턴스 생성
train_dataset = TextDataset('train.csv', tokenizer, vocab, max_length=256, label_transform=label_transform)
test_dataset = TextDataset('test.csv', tokenizer, vocab, max_length=256, label_transform=label_transform)

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes, num_filters, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [Batch size, Word count, Embedding dim]
        x = x.unsqueeze(1)  # [Batch size, Channel(1), Word count, Embedding dim]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# 모델 인스턴스 생성
vocab_size = len(vocab)
embed_dim = 300  # 임베딩 차원
num_classes = len(label_encoder.classes_)  # 레이블 개수
filter_sizes = [3, 4, 5]
num_filters = 100
dropout = 0.5

model = TextCNN(vocab_size, embed_dim, num_classes, filter_sizes, num_filters, dropout)


import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)


import matplotlib.pyplot as plt

# 훈련 함수 정의
def train(model, iterator, optimizer, criterion):
    model.train()  # 모델을 훈련 모드로 설정
    epoch_loss = 0

    for batch in iterator:
        texts, labels = batch
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()  # 그래디언트 초기화
        predictions = model(texts)  # 모델로부터 예측값 계산
        loss = criterion(predictions, labels)  # 손실 계산
        loss.backward()  # 역전파 수행
        optimizer.step()  # 옵티마이저로 파라미터 업데이트
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)  # 에포크 손실 평균 반환

# 검증 함수 정의
def evaluate(model, iterator, criterion):
    model.eval()  # 모델을 평가 모드로 설정
    epoch_loss = 0

    with torch.no_grad():
        for batch in iterator:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)

            predictions = model(texts)  # 모델로부터 예측값 계산 (평가 시 그래디언트 계산 X)
            loss = criterion(predictions, labels)  # 손실 계산
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)  # 에포크 손실 평균 반환

# 훈련 및 검증 손실을 기록할 리스트 초기화
train_losses = []
valid_losses = []

num_epochs = 3  # 훈련 에포크 수
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)  # 훈련 함수 호출하여 훈련 손실 계산
    valid_loss = evaluate(model, test_loader, criterion)  # 검증 함수 호출하여 검증 손실 계산

    train_losses.append(train_loss)  # 훈련 손실 기록
    valid_losses.append(valid_loss)  # 검증 손실 기록

    # 에포크별로 훈련 및 검증 손실 출력
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}')

# 손실 그래프 시각화
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')  # 훈련 손실 그래프
plt.plot(valid_losses, label='Validation Loss')  # 검증 손실 그래프
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_performance(model, iterator, criterion):
    # 예측 및 실제 레이블 저장을 위한 리스트 초기화
    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            # 예측 결과를 CPU로 이동하고 넘파이 배열로 변환
            predictions = torch.argmax(predictions, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            # 리스트에 추가
            all_predictions.extend(predictions)
            all_labels.extend(labels)

    # 성능 지표 계산
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return accuracy, precision, recall, f1

# 성능 평가
accuracy, precision, recall, f1 = evaluate_performance(model, test_loader, criterion)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# 예측하고자 하는 텍스트
text_to_predict = "I am feeling very happy today!"


# 텍스트 전처리 (토큰화, 수치화, 패딩)
tokenized_text = tokenizer(text_to_predict)
numericalized_text = [vocab.get_stoi()[token] if token in vocab.get_stoi() else vocab.get_stoi()['<unk>'] for token in tokenized_text]

# 최소 길이를 설정합니다. 이는 모델의 컨볼루션 레이어 필터 크기에 따라 달라질 수 있습니다.
min_length = 5  # 예를 들어, 컨볼루션 필터 크기의 최대값

# 패딩 추가
padded_text = pad_sequence([torch.tensor(numericalized_text)], padding_value=vocab.get_stoi()['<pad>'], batch_first=True)
if padded_text.size(1) < min_length:
    # 길이가 최소 길이보다 짧은 경우, 추가 패딩을 적용
    padding = torch.full((padded_text.size(0), min_length - padded_text.size(1)), vocab.get_stoi()['<pad>'], dtype=torch.long)
    padded_text = torch.cat([padded_text, padding], dim=1)

# 모델을 평가 모드로 설정하고 예측 수행
model.eval()
with torch.no_grad():
    prediction = model(padded_text.to(device))
    predicted_label = torch.argmax(prediction, dim=1).item()

# 예측된 레이블을 문자열로 변환
predicted_label_str = label_encoder.inverse_transform([predicted_label])[0]
print(f"Predicted label: {predicted_label_str}")
