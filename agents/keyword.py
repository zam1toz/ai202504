from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL
import json

# GPT 기반 검색 키워드 생성 에이전트
# 음식 또는 활동 추천 결과를 바탕으로 장소 검색에 적합한 키워드를 추출합니다.
llm = ChatOpenAI(
    model=OPENAI_MODEL,                     # 사용할 GPT 모델
    api_key=OPENAI_API_KEY,                # 환경변수에서 불러온 OpenAI API 키
    temperature=0.3,                       # 창의성 낮게 (정확성 위주)
    model_kwargs={
        "response_format": {"type": "json_object"}  # 응답 형식 강제: JSON 객체
    }
)

def generate_search_keyword(state: dict) -> dict:
    """
    GPT를 사용하여 추천 항목을 바탕으로 장소 검색용 키워드를 생성하는 함수입니다.
    예: 김치찌개 → 한식, 책 읽기 → 북카페
    """

    # 추천 항목 리스트 추출 (음식 또는 활동)
    items = state.get("recommended_items", ["추천"])
    if isinstance(items, dict):
        # 딕셔너리인 경우 → 값만 추출 (중첩 flatten)
        items = [i for sub in items.values() for i in (sub if isinstance(sub, list) else [sub])]
    elif not isinstance(items, list):
        items = [str(items)]  # 문자열인 경우 → 리스트로 변환

    item = items[0]  # 첫 번째 추천 항목을 기반으로 키워드 생성

    user_input = state.get("user_input", "")      # 사용자 입력
    intent = state.get("intent", "food")          # food 또는 activity

    # GPT 프롬프트 작성
    prompt = f"""사용자의 입력: "{user_input}"
추천 항목: "{item}"
의도: "{intent}"

이 항목을 장소에서 검색하려고 합니다.
음식이라면 음식 종류(예: 김치찌개 → 한식),
활동이라면 장소 유형(예: 책 읽기 → 북카페)으로 변환하세요.

결과는 반드시 JSON 배열로 출력하세요.
예: ["한식"]
"""  # f-string 끝

    # GPT 호출
    response = llm.invoke([{"role": "user", "content": prompt.strip()}])

    # GPT 응답 파싱
    keywords = json.loads(response.content)

    # dict 형태 응답 → 값 추출
    if isinstance(keywords, dict):
        keywords = [i for sub in keywords.values() for i in (sub if isinstance(sub, list) else [sub])]
    elif not isinstance(keywords, list):
        keywords = [str(keywords)]

    # 생성된 키워드 중 첫 번째를 상태에 추가
    return {**state, "search_keyword": keywords[0] if keywords else item}
    