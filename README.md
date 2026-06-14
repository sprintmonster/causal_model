# 인과 추론 요약

## 프로젝트 목적

신문 OCR 텍스트에서 추출한 지식그래프를 바탕으로 인과모델을 만들고,1920년 1월 17일에 시행된 금주법(Prohibition) 관련 사건을 어떤 요인이 이끌었는지 분석하는 것이 목적입니다.

## 데이터와 파이프라인

- 원천 데이터: `data/1916` ~ `data/1920` 아래의 OCR 텍스트 파일
- 추출 단계: 하루 단위로 지식 그래프를 생성한 뒤 병합하여 지식그래프 추출( RTX 4090, Qwen 2.5 7B LLM을 사용 )
- 그래프 모델: `CAUSES`, `ENABLES`, `MEDIATES` 관계를 사용하는 DAG 형태의 확률적 인과모델
- 추론 단계: noisy-OR 전파와 `see(X)` 개입 랭킹

## 주요 결과물

- 병합된 인과 그래프: `final_graph/final_causal_graph.json`
- 인과모델: `final_graph/causal_model.json`
- 개입 랭킹 리포트: `final_graph/causal_inference_report.json`
- 금주법 시행 리포트: `final_graph/prohibition_enactment_inference_report.json`
- 시각화 결과: `final_graph/prohibition_cluster_dag_typed.png`

## 금주법 정책 클러스터 분석

<img width="3943" height="2620" alt="prohibition_cluster_dag_typed" src="https://github.com/user-attachments/assets/4a931b37-10ef-4412-9495-c7ce4ae24fb4" />

### 개요

금주법은 입법 논의, 비준, 집행, 전시 통제, 사법 판단에 의해 영향을 받은 것으로 해석됩니다.

### 1단계: 입법 추진

#### 핵심 흐름

- `event:prohibition-bill-discussion` -> `claim:prohibition-success`
- `event:sheppard_opinion` -> `event:passage_of_district_prohibition_bill`
- `event:senate-passes-prohibition-bill` -> `topic:district-prohibition-bill`

#### 해석

금주법은 찬반 논쟁과 법안 심의가 축적되면서 통과된 흐름으로 보입니다.

### 2단계: 제도적 기반

#### 핵심 흐름

- `event:nebraska_ratification` -> `claim:prohibition_amendment_ratified`

#### 해석

금주가 단순한 지역 조치가 아니라, 더 큰 헌법·연방 차원의 제도화 과정에 들어가 있었음을 보여줍니다.

### 3단계: 시행 및 집행

#### 핵심 흐름

- `event:supreme-court-decision` -> `event:prohibition-enforcement-act`
- `event:prohibition_agents_arrested`
- `event:prohibition_of_spirits`

#### 해석

법이 생긴 뒤에는 법 집행과 단속이 중심이 되었습니다.

### 4단계: 전시 통제 및 국가 규율

#### 핵심 흐름

- `event:war_prohibition_discussion`
- `event:war-time-prohibition-repeal`
- `event:president-end-war-prohibition`

#### 해석

금주법이 전쟁기 절제, 국가 통제, 도덕 개혁과 연결된 정책으로 작동했음을 보여줍니다.

### 데이터 규모

| 항목 | 값 |
|------|-----|
| 노드 | 84개 |
| 엣지 | 22개 |
| 구성 | prohibition 관련 seed 기반 |

### 결론

금주법이 시행된 이유는 **도덕적 개혁 + 입법 추진 + 비준 + 집행 + 전시 통제**가 동시에 작동한 결과로 해석할 수 있습니다.

> 그래프의 중심은 **금주법 자체**보다 **금주법을 둘러싼 정치·정책 담론**입니다.

## 주요 인과추론 결과

주요 개입 랭킹에서 사용한 타깃은 다음과 같습니다.

- `event:prohibition_convention`

기준 확률:

- $P(\text{target}) = 0.001605$

직접 부모 노드:

- `event:president_work` - President Wilson's Work
- `event:ratification_of_treaty` - Ratification of treaty
- `event:subnormal_survey` - Survey of Subnormal Minds in Cook County
- `event:war_operations` - War Operations in Caucasus and Persia

가장 큰 개입 효과:

- `event:president_work`
- `see(parent=1.0)` 시 $P(\text{target})$가 `0.001605`에서 `0.002602`로 증가
- 절대 변화량: `0.000997`

## 금주법 시행 결과

타깃:

- `event:prohibition-enforcement-act`

기준 확률:

- $P(\text{target}) = 0.001002$

직접 부모:

- `event:supreme-court-decision` - Supreme Court Decision

효과:

- `See(parent=1.0)` 시 $P(\text{target})$가 `0.001002`에서 `0.001999`로 증가
- 절대 변화량: `0.000997`

## 해석

이 그래프는 금주법 관련 결과가 단일 원인으로 생긴 것이 아니라, 입법, 사법, 전시, 행정이 결합된 클러스터의 결과로 나타난다는 점을 보여줍니다.

다만 이 모델은 엄밀한 통계적 인과효과 추정이라기보다, 구조화된 텍스트 증거로부터 만든 **graph-based causal hypothesis model**로 보는 것이 적절합니다.

## 한계

- 엣지 확률은 실험 데이터가 아니라 추출된 support statistics에서 나온 값입니다.
- 시행 관련 영역은 아직 희소해서 direct parent가 1개인 타깃도 있습니다.
- 결과는 최종적인 인과 증명이라기보다 구조화된 인과 가설로 해석해야 합니다.

