#1. 필요한 라이브러리 설치 및 임포트**

#!pip install konlpy==0.6.0
#!pip install pandas ktextaug

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from konlpy.tag import Okt
import pandas as pd
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from ktextaug import TextAugmentation

#! git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git

#cd Mecab-ko-for-Google-Colab/

#!bash install_mecab-ko_on_colab_light_220429.sh

#2. 데이터셋 불러오기 및 텍스트 정제**

import os
file_list = os.listdir('/content/sample_data/')
print(file_list)

df = pd.read_csv('/content/sample_data/DataSet.csv', encoding='utf-8')

# 텍스트 정제 함수 정의
def clean_text(text):
    text = re.sub(r"[^가-힣\s]", "", text)  # 한글과 공백을 제외하고 모두 제거
    text = re.sub(r"\s+", " ", text)  # 연속된 공백은 하나의 공백으로
    return text.strip()

import pandas as pd
import MeCab

# MeCab-ko 초기화
mecab = MeCab.Tagger()

# 데이터셋 불러오기
df = pd.read_csv('/content/sample_data/DataSet.csv', encoding='utf-8')

# TextAugmentation 객체 초기화
augmenter = TextAugmentation(tokenizer=mecab)

augmented_texts = []
augmented_categories = []

for _, row in df.iterrows():
    try:
        augmented_text = augmenter.generate(row['Title'])
        augmented_texts.append(augmented_text)
        augmented_categories.append(row['Category'])
    except Exception as e:
        print(f"Error in text augmentation: {e}")

augmented_df = pd.DataFrame({'Category': augmented_categories, 'Title': augmented_texts})
full_df = pd.concat([df, augmented_df]).reset_index(drop=True)


#**3. 불용어 제거 및 텍스트 전처리**

# 불용어 정의
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

# 불용어 제거 함수 정의
okt = Okt()
def remove_stopwords(text):
    tokenized_text = okt.morphs(text, stem=True)
    non_stopwords = [word for word in tokenized_text if word not in stopwords]
    return ' '.join(non_stopwords)

# 텍스트 전처리
df['cleaned_text'] = df['Title'].apply(clean_text).apply(remove_stopwords)

# 텍스트 토큰화
def tokenize_text(text):
    return okt.morphs(text)

df['tokenized_text'] = df['cleaned_text'].apply(tokenize_text)

#**4. 단어 빈도수 계산 및 단어 사전 생성**

# 단어 빈도수 계산 및 단어 사전 생성
word_counts = Counter()
df['tokenized_text'].apply(lambda tokens: word_counts.update(tokens))
vocab = {word: index + 2 for index, (word, _) in enumerate(word_counts.most_common())}
vocab['<pad>'] = 0
vocab['<unk>'] = 1

#**5. 레이블 인코딩 및 데이터셋 클래스 정의**

from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

# 레이블 인코딩
label_encoder = LabelEncoder()
df['encoded_labels'] = label_encoder.fit_transform(df['Category'])

# 데이터 분할
x = df['cleaned_text'].apply(lambda x: [vocab.get(word, vocab['<unk>']) for word in x.split()]).tolist()
y = df['encoded_labels'].tolist()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# 데이터셋 클래스 정의
class TextDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return torch.tensor(self.x_data[idx], dtype=torch.long), torch.tensor(self.y_data[idx], dtype=torch.long)

# collate_fn 함수를 클래스 외부에 정의
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, padding_value=vocab['<pad>'], batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts, labels

# 데이터셋 인스턴스 및 데이터 로더 생성
train_dataset = TextDataset(x_train, y_train)
test_dataset = TextDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


#**6. LSTM 모델 클래스 정의 및 초기화**

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional  # 속성 초기화 추가

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, _) = self.lstm(embedded)
        if self.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden.squeeze(0))

# 이 변수들은 모델 인스턴스를 생성하기 전에 정의되어야 합니다.
INPUT_DIM = len(vocab)
EMBEDDING_DIM = 100  # 예: 100
HIDDEN_DIM = 256    # 예: 256
OUTPUT_DIM = len(label_encoder.classes_)  # 클래스의 수
N_LAYERS = 2        # 예: 2
BIDIRECTIONAL = True
DROPOUT = 0.5       # 예: 0.5

# 모델 인스턴스 생성
model = LSTMClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)


#**7. 손실 함수 및 옵티마이저 설정, 데이터 로더 생성**


# 손실 함수 및 옵티마이저 설정
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


#**8. 훈련 및 평가 함수 정의, 모델 훈련 및 평가**

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch in iterator:
        optimizer.zero_grad()
        text, labels = batch
        predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct += (predictions.argmax(1) == labels).sum().item()
        total += labels.size(0)

    epoch_acc = correct / total
    return epoch_loss / len(iterator), epoch_acc

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            predictions = model(text)
            loss = criterion(predictions, labels)

            epoch_loss += loss.item()
            correct += (predictions.argmax(1) == labels).sum().item()
            total += labels.size(0)

    epoch_acc = correct / total
    return epoch_loss / len(iterator), epoch_acc


# 훈련 및 평가
N_EPOCHS = 20
train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')



#**9. 결과 시각화**

# 손실 그래프 그리기
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training accuracy')
plt.plot(test_accuracies, label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

def preprocess_text(text, okt, vocab, max_length):
    # 텍스트 정제 및 토큰화
    cleaned_text = clean_text(text)
    tokenized_text = okt.morphs(cleaned_text, stem=True)
    non_stopwords = [word for word in tokenized_text if word not in stopwords]

    # 토큰을 숫자로 변환
    numericalized_text = [vocab.get(token, vocab['<unk>']) for token in non_stopwords]

    # 리스트를 Tensor로 변환
    numericalized_tensor = torch.tensor(numericalized_text, dtype=torch.long)

    # 패딩 적용
    padded_tensor = pad_sequence([numericalized_tensor], padding_value=vocab['<pad>'], batch_first=True)

    # 첫 번째 요소 반환 (배치 크기 1)
    return padded_tensor[0]


def classify_sentence(sentence, model, okt, vocab, max_length):
    # 문장 전처리
    preprocessed_text = preprocess_text(sentence, okt, vocab, max_length)

    # 모델 입력을 위한 텐서 변환
    # input_tensor = torch.tensor([preprocessed_text]) # 오류가 발생하는 부분
    input_tensor = preprocessed_text.unsqueeze(0) # 수정된 부분

    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)
        # 차원 확인 후 적절한 dim 값 설정
        if prediction.ndim == 1:
            predicted_label = torch.argmax(prediction)  # 단일 차원
        else:
            predicted_label = torch.argmax(prediction, dim=1)  # 배치 차원 포함

    return label_encoder.inverse_transform([predicted_label.item()])[0]




# ipywidgets 라이브러리 설치
#!pip install ipywidgets

import ipywidgets as widgets
from IPython.display import display

# 텍스트 입력 위젯 생성
text_input = widgets.Textarea(
    value='',
    placeholder='Enter text here',
    description='Text:',
    disabled=False
)

# 버튼 위젯 생성
predict_button = widgets.Button(
    description='Classify',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click to classify the text'
)

# 출력 위젯 생성
output = widgets.Output()

MAX_LENGTH = 30

# 버튼 클릭 이벤트 핸들러
def on_predict_button_clicked(b):
    # 입력된 텍스트 가져오기
    input_text = text_input.value


    predicted_category = classify_sentence(input_text, model, okt, vocab, MAX_LENGTH)

    # 결과 출력
    with output:
        output.clear_output()
        print(f"Input Text: '{input_text}'\nPredicted Category: {predicted_category}")

# 버튼 이벤트에 핸들러 연결
predict_button.on_click(on_predict_button_clicked)

# 위젯 표시
display(text_input, predict_button, output)
