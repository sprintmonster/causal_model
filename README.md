<img width="3943" height="2620" alt="prohibition_cluster_dag_typed" src="https://github.com/user-attachments/assets/92b28ab9-9ff8-4ca3-987f-fc2f6c04642e" />

그래프를 바탕으로 보면, 금주법은 단일 사건이 아니라 입법 논의, 비준, 집행, 전시 통제, 사법 판단이 함께 얽힌 정책 클러스터로 해석됩니다.

핵심 해석은 이렇습니다.

입법이 먼저 움직였습니다.

event:prohibition-bill-discussion -> claim:prohibition-success
event:sheppard_opinion -> event:passage_of_district_prohibition_bill
event:senate-passes-prohibition-bill -> topic:district-prohibition-bill
즉, 금주법은 찬반 논쟁과 법안 심의가 축적되면서 통과된 흐름으로 보입니다.
주별 비준과 헌법 개정이 제도적 기반이었습니다.

event:nebraska_ratification -> claim:prohibition_amendment_ratified
이건 금주가 단순한 지역 조치가 아니라, 더 큰 헌법·연방 차원의 제도화 과정에 들어가 있었음을 보여줍니다.
시행은 사법 판단과 집행 체계로 이어졌습니다.

event:supreme-court-decision -> event:prohibition-enforcement-act
event:prohibition_agents_arrested, event:prohibition_of_spirits
즉, 법이 생긴 뒤에는 법 집행과 단속이 중심이 되었습니다.
전시 분위기와 국가 규율 논리도 함께 작동했습니다.

event:war_prohibition_discussion
event:war-time-prohibition-repeal
event:president-end-war-prohibition
금주법이 전쟁기 절제, 국가 통제, 도덕 개혁과 연결된 정책으로 읽힙니다.
그래프의 중심은 ‘금주법 자체’보다 ‘금주법을 둘러싼 정치·정책 담론’입니다.

현재 클러스터 DAG는 84개 노드, 22개 엣지로 구성돼 있고, prohibition 관련 seed들을 묶어 만든 구조입니다.
그래서 “왜 시행됐는가”를 보면, 도덕적 개혁 + 입법 추진 + 비준 + 집행 + 전시 통제가 합쳐진 결과로 해석하는 게 가장 자연스럽습니다.
