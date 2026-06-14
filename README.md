<img width="3943" height="2620" alt="prohibition_cluster_dag_typed" src="https://github.com/user-attachments/assets/978771ee-0907-4b1a-9d2b-f7a94436f674" />


# 금주법(Prohibition) 정책 클러스터 분석

## 개요
금주법은 **단일 사건이 아니라 정책 클러스터**로, 입법 논의, 비준, 집행, 전시 통제, 사법 판단이 함께 얽힌 복합 정책 체계입니다.

---

## 1단계: 입법 추진

### 핵심 흐름
- `event:prohibition-bill-discussion` → `claim:prohibition-success`
- `event:sheppard_opinion` → `event:passage_of_district_prohibition_bill`
- `event:senate-passes-prohibition-bill` → `topic:district-prohibition-bill`

### 해석
금주법은 **찬반 논쟁과 법안 심의가 축적**되면서 통과된 흐름으로 보입니다.

---

## 2단계: 제도적 기반 (헌법 개정 & 비준)

### 핵심 흐름
- `event:nebraska_ratification` → `claim:prohibition_amendment_ratified`

### 해석
금주가 단순한 지역 조치가 아니라 **더 큰 헌법·연방 차원의 제도화 과정**에 들어가 있었음을 보여줍니다.

---

## 3단계: 시행 및 집행

### 핵심 흐름
- `event:supreme-court-decision` → `event:prohibition-enforcement-act`
- `event:prohibition_agents_arrested`
- `event:prohibition_of_spirits`

### 해석
법이 생긴 뒤에는 **법 집행과 단속**이 중심이 되었습니다.

---

## 4단계: 전시 통제 & 국가 규율

### 핵심 흐름
- `event:war_prohibition_discussion`
- `event:war-time-prohibition-repeal`
- `event:president-end-war-prohibition`

### 해석
금주법이 **전쟁기 절제, 국가 통제, 도덕 개혁**과 연결된 정책으로 작동했습니다.

---

## 데이터 규모

| 항목 | 값 |
|------|-----|
| 노드 | 84개 |
| 엣지 | 22개 |
| 구성 | prohibition 관련 seed 기반 |

---

## 결론

금주법이 시행된 이유는:

**도덕적 개혁 + 입법 추진 + 비준 + 집행 + 전시 통제**가 동시에 작동한 결과

> 그래프의 중심은 **'금주법 자체'보다 '금주법을 둘러싼 정치·정책 담론'**입니다.
