import pandas as pd, numpy as np

#DataFrame : 2차원 리스트 매개변수
a = pd.DataFrame([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print(a, type(a))
#Series - 1차원 데이터 다룰때 사용
s = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0])
print(s, type(s))

#키, 몸무게, 유형 데이터프레임 생성
tbl = pd.DataFrame({
    "weight":[80.0, 70.4, 65.5, 45.9, 51.2],
    "height":[170, 180, 155, 143, 154],
    "type":["f", "n", "n", "t", "t"]
})

print(tbl[["weight","height"]])
print(tbl[2:4]) 
print("-- height 가 160이상인것")
print(tbl[tbl.height >= 160])
print("-- type이 f인것")
print(tbl[tbl.type =="f"])
print("-- 정렬")
print(tbl.sort_values(by="weight", ascending=False))
print("-- 행과 열 반전")
print(tbl.T)