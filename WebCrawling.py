import requests
from bs4 import BeautifulSoup
import csv
import time

# 카테고리와 선택자 정보
categories = {
    "정치": "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100",
    "경제": "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=101",
    "IT/과학": "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=105",
    "사회": "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=102",
    "생활/문화" : "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=103",
    "세계" : "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=104",
    "연예": "https://entertain.naver.com/home?page=",
    "스포츠": "https://sports.news.naver.com/index?page=",
    
}

with open('./Dataset/DataSet.csv', 'w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(['Category', 'Title'])
    
    for category, base_url in categories.items():
        for page in range(1, 77):  
            url = base_url + "&page=" + str(page)  
            try:
                raw = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                raw.raise_for_status()  
                html = BeautifulSoup(raw.text, "html.parser")

                if category == "스포츠":  # label을 category로 변경
                    selector = "strong.title"
                elif category == "연예":  # label을 category로 변경
                    selector = "a.title"
                else:    
                     selector = "a.sh_text_headline"
                    
            
                articles = html.select(selector)

                for ar in articles:
                    title = ar.text.strip() 
                    writer.writerow([category, title])  

                time.sleep(1)  # 각 요청 후 1초 대기
            except requests.RequestException as e:
                print(f"Request error for URL: {url} - {e}")
            except Exception as e:
                print(f"Error processing page: {url} - {e}")