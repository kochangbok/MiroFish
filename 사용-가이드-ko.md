# MiroFish 사용 가이드 (한국어)

이 문서는 **MiroFish를 로컬에서 실행하고, 문서를 넣어 Step1~Step5 흐름으로 사용하는 방법**을 정리한 안내서입니다.

## 1. MiroFish로 할 수 있는 일

MiroFish는 PDF/MD/TXT 같은 문서를 입력으로 받아,

1. **온톨로지/지식그래프 생성**
2. **에이전트 페르소나와 시뮬레이션 환경 구성**
3. **Twitter/Reddit 스타일 병렬 시뮬레이션 실행**
4. **결과 보고서 생성**
5. **Report Agent 및 개별 에이전트와 대화/설문**

까지 이어지는 멀티에이전트 시뮬레이션 도구입니다.

---

## 2. 사전 준비

### 필수 런타임
- Node.js 18+
- Python 3.11 또는 3.12
- `uv`
- Zep API Key

### 선택지 A: 일반 OpenAI 호환 API 사용
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL_NAME`

### 선택지 B: 로컬 Codex OAuth 브리지 사용
이 작업 디렉터리에는 `../codex-bridge`가 준비되어 있습니다.
이 방식은 **로컬에 로그인된 Codex/ChatGPT 세션**을 MiroFish에 연결하는 실험용 브리지입니다.

추가 준비물:
- `codex` CLI
- `codex login` 완료 상태

---

## 3. `.env` 설정

프로젝트 루트에서:

```bash
cp .env.example .env
```

### 3-1. 일반 OpenAI 호환 API 예시

```env
LLM_API_KEY=sk-your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini
ZEP_API_KEY=your-zep-api-key
```

### 3-2. Codex OAuth 브리지 예시

```env
LLM_API_KEY=local-codex-bridge
LLM_BASE_URL=http://127.0.0.1:8787/v1
LLM_MODEL_NAME=gpt-5.1-codex-mini
ZEP_API_KEY=your-zep-api-key
```

> 참고: 브리지 방식에서는 `LLM_API_KEY`가 사실상 더미값이어도 됩니다.

---

## 4. 실행 방법

### 4-1. Python / 의존성 준비

```bash
pyenv install 3.12.9   # 아직 없을 때만
pyenv local 3.12.9
brew install uv        # 아직 없을 때만
```

### 4-2. 프로젝트 의존성 설치

```bash
npm install
npm run setup:all
```

---

## 5. Codex OAuth 브리지 방식으로 실행할 때

브리지 프로젝트는 상위 디렉터리의 `../codex-bridge`에 있습니다.

### 5-1. 로그인 상태 확인

```bash
codex login status
```

### 5-2. 브리지 실행

```bash
cd ../codex-bridge
npm install
PORT=8787 \
CODEX_MODEL=gpt-5.1-codex-mini \
CODEX_BRIDGE_WORKDIR=/Users/george/.superset/projects/mirofishi-test/MiroFish \
npm start
```

### 5-3. 헬스체크

```bash
curl -s http://127.0.0.1:8787/health
```

정상이라면 브리지가 살아 있고, MiroFish는 `LLM_BASE_URL=http://127.0.0.1:8787/v1`로 붙을 수 있습니다.

---

## 6. MiroFish 앱 실행

프로젝트 루트에서:

### 백엔드
```bash
npm run backend
```

### 프론트엔드
```bash
npm run frontend
```

브라우저 접속:
- `http://localhost:3000`

---

## 7. 화면 사용 흐름

## Step 1. 그래프 구축
- 문서(PDF/MD/TXT) 업로드
- 시뮬레이션 프롬프트 입력
- 온톨로지 생성 → GraphRAG/지식그래프 구축

이 단계가 끝나면 엔티티/관계/메모리 기반이 만들어집니다.

## Step 2. 환경 구성
- 에이전트 페르소나 생성
- 시뮬레이션 시간/라운드/플랫폼 설정 생성
- 추천 알고리즘/서사 방향/초기 화제 구성

처음에는 **라운드 수를 작게** 잡는 것이 좋습니다.

## Step 3. 시뮬레이션 시작
- 두 플랫폼이 병렬로 움직입니다.
- 게시/댓글/리포스트/검색/팔로우 같은 행동 로그가 누적됩니다.
- 그래프도 함께 갱신됩니다.

## Step 4. 보고서 생성
- Report Agent가 여러 도구를 호출해 결과를 정리합니다.
- 핵심 사실, 관계, 엔티티, 인터뷰 결과를 구조화해 보여줍니다.

## Step 5. 심층 상호작용
- Report Agent와 후속 대화
- 개별 에이전트 인터뷰
- 여러 에이전트 대상 설문 발송

---

## 8. 처음 해보기 좋은 입력 예시

### 예시 1: 공지문 반응
- 파일: 공지문 초안
- 프롬프트: `이 공지문을 올리면 사용자 여론이 어떻게 변할지 시뮬레이션해줘.`

### 예시 2: 정책 변경 반응
- 파일: 정책/약관 변경 내용
- 프롬프트: `정책 변경 발표 이후 커뮤니티 반응과 논쟁 포인트를 예측해줘.`

### 예시 3: 사건 전파 분석
- 파일: 뉴스 요약, 내부 메모, SNS 캡처 요약
- 프롬프트: `이 이슈가 어떤 집단을 통해 확산되는지 시뮬레이션해줘.`

---

## 9. 비용/실행 팁

- MiroFish는 단일 챗봇 호출보다 **LLM 호출량이 많습니다**.
- 문서가 길고, 라운드가 많고, 에이전트가 많을수록 비용이 빠르게 올라갑니다.
- 첫 실행은 아래처럼 권장합니다.

### 권장 시작값
- 문서 1개
- 짧은 텍스트 또는 작은 PDF
- 라운드 수 10~20 정도
- 보고서 생성까지 한 번만 확인

---

## 10. 문제 해결

### 프론트는 뜨는데 데이터가 안 보일 때
- 백엔드가 켜져 있는지 확인
- `http://localhost:5001` 응답 여부 확인
- 브라우저 콘솔에서 `ERR_CONNECTION_REFUSED`가 있으면 백엔드 미실행 가능성이 큼

### Zep 관련 오류가 날 때
- `.env`의 `ZEP_API_KEY` 확인
- Zep Cloud 계정/키 상태 확인

### LLM 호출이 실패할 때
- 일반 API 방식이면 `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_NAME` 확인
- 브리지 방식이면 `codex login status`와 `http://127.0.0.1:8787/health` 확인

### 시뮬레이션이 너무 무겁거나 느릴 때
- 라운드 수 줄이기
- 문서 수/길이 줄이기
- 먼저 Step1~Step3까지만 확인 후 보고서 생성

---

## 11. 현재 이 작업 디렉터리 기준 참고 경로

- MiroFish 앱: `/Users/george/.superset/projects/mirofishi-test/MiroFish`
- Codex OAuth 브리지: `/Users/george/.superset/projects/mirofishi-test/codex-bridge`
- 브리지 사용 설명: `../codex-bridge/사용방법.md`

---

## 12. 한 줄 요약

- **정석 방식**: OpenAI 호환 API + Zep 키로 실행
- **실험 방식**: 로컬 Codex OAuth 브리지 + Zep 키로 실행
- **추천 사용 순서**: 작은 문서 → 적은 라운드 → Step1~Step5 순서 검증
