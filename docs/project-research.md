# AI Buddy – Project Research (v0.1)

> Scope: Global, consumer mobile (iOS + Android), edge-first multimodal assistant for travelers. This document consolidates market analysis, competitor scan, gap analysis, feasibility, architecture options, privacy/compliance, risks, KPIs, and a research plan.

---

## 1) Problem, Opportunity & Thesis

**Core problem**: Travelers struggle with _situational_ tasks (finding the way, ordering safely, asking for help) because current tools are _translation-centric_ rather than _goal-centric_. They translate text/audio but don’t _guide_.

**Opportunity**: The travel & tourism sector rebounded strongly. UN Tourism estimates ~**1.4B** international arrivals in **2024** (≈99% of 2019 levels), with Q1 **2025** +5% YoY, exceeding 2019 by ~3% [UN Tourism Barometer, Jan/May 2025]. WTTC’s EIR reports **$10.9T** GDP contribution (**10%** of global GDP) for **2024**—a new record—and **357M** jobs [WTTC 2024/2025]. This rising tide + AI-on-device momentum creates space for a **contextual travel buddy**.

**Thesis**: Build a **multimodal, goal-oriented AI buddy** that _sees_ (camera OCR), _locates_ (GPS+maps), _listens & speaks_ (ASR/TTS), _reasons_ (LLM), and _guides_ users step‑by‑step—_offline-first_ with cloud enhancement.

**Primary jobs-to-be-done**

-   "I’m lost—help me get back to my hotel/home."
-   "I’m hungry—help me understand this menu and order safely."
-   "I’m shopping—help me check ingredients/warranty/price and decide."
-   "I need help—tell me who to ask and what to say (politely)."

---

## 2) Users, Personas & Contexts

**Personas**

1. **Tourist** (short-stay, new city): needs wayfinding, menus, transit.
2. **Business traveler**: fast interactions, reliable offline, airports, taxis.
3. **Expat/student**: day‑to‑day shopping, bureaucratic forms, banks.
4. **Service staff** (secondary): quick two-way phrases with customers.

**Contexts**: street/transit, airport/train, restaurant/cafe, retail aisle, hotel lobby, emergencies.

---

## 3) Market & Trend Snapshot (selected stats)

-   **International arrivals**: ~**1.4B** in 2024; Q1 2025 +5% YoY (UN Tourism World Tourism Barometer, Jan & May 2025).
-   **Sector size**: **$10.9T** GDP in 2024; ~**10%** of global economy; **357M** jobs (WTTC EIR 2024/2025).
-   **Destination example**: Spain recorded **94M** international tourists in 2024 (record high), highlighting sustained demand for culinary/cultural travel.
-   **Platform momentum**: System-level translation/vision features are expanding (e.g., Android **Scroll & translate** continuous on-screen translation; Apple Vision/Live Text).

> Implication: Large, resilient demand + platform primitives → _now_ is a good time for a guidance‑first experience.

**Sources**: UN Tourism Barometer Jan/May 2025; WTTC EIR 2024/2025; AP Spain 2024 record; Android Circle-to-Search Scroll & translate (links in §15).

---

## 4) Competitive Landscape (apps & devices)

### 4.1 Translation Apps (consumer)

| Product                  | Strengths                                                                           | Notable Features                                                                                                                                                                            | Limitations vs. Buddy Concept                                                                                    |
| ------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Google Translate**     | Massive language coverage; strong camera + conversation; offline packs              | App Store lists up to **249** languages; instant camera translation (~**94** languages); offline (~**59**); conversation (70+); **Scroll & translate** for continuous on-screen translation | Translation‑centric; minimal goal memory; no stepwise guidance, limited diet/safety context                      |
| **Microsoft Translator** | Group/multi-device conversation; split-screen; offline packs (feature set evolving) | Conversations, text/voice/camera modes; phrasebook                                                                                                                                          | Reports in 2025 indicate changes/limits to offline on Android; still translation‑first, not situational guidance |
| **DeepL**                | High quality for many European languages; clean UX                                  | Mobile apps with text translation & TTS                                                                                                                                                     | Narrower language set; camera/menu workflows less mature; not guidance-oriented                                  |
| **Naver Papago**         | Good Asian language support; simple camera & voice                                  | ~14 languages; mobile + web                                                                                                                                                                 | Smaller coverage; no contextual guidance layer                                                                   |
| **iTranslate**           | Camera mode; offline; phrasebooks                                                   | 100+ languages; offline mode                                                                                                                                                                | Paywalls; still translation‑first; limited situational reasoning                                                 |

### 4.2 Dedicated Translator Devices

| Device             | Value                                                                           | Notable                                           | Limitations                                                                                |
| ------------------ | ------------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Timekettle T1**  | Handheld with **31+ offline language pairs**, photo translation; 2‑year roaming | Edge small model; offline translation; camera OCR | Pairs ≠ languages; mixed camera quality; extra device to carry; limited beyond translation |
| **Pocketalk S/S2** | Two‑way voice across **92+** languages; camera translation; eSIM options        | Works without phone; noise‑cancel mics            | Additional hardware; primarily translation; limited guidance/integration                   |

**Takeaway**: The field is crowded with _translators_. **No major player is a _situational buddy_** that fuses camera + maps + conversation + safety + dietary reasoning into a single, goal‑oriented flow.

**Key platform primitives we can leverage instead of reinventing**:

-   **OCR**: Google ML Kit Text Recognition v2 (Android), Apple Vision/VisionKit Live Text (iOS).
-   **ASR**: On‑device Whisper‑class (quantized), Apple on‑device speech APIs.
-   **MT**: Distilled multilingual NMT (e.g., Meta **NLLB-200**) for broad coverage.
-   **TTS**: Apple `AVSpeechSynthesizer` / Android TTS engines; cloud neural TTS for premium voices.

---

## 5) Gap Analysis → Differentiators

**Observed gaps in current apps/devices**

1. **No goal memory**: Apps translate but don’t remember “get me back to my hotel” across turns.
2. **Weak situational fusion**: Camera text, GPS position, map POIs, and dialog aren’t fused to _guide_.
3. **Restaurant intelligence**: Few apps explain dishes, map to ingredients/allergens/diet preferences, and generate polite localized order phrases.
4. **Proactive help**: No suggestions like “turn around for metro entrance,” “ask the info desk,” or “this dish conflicts with your allergy.”
5. **Offline beyond translation**: Offline camera+voice works, but **offline reasoning & guidance** is rare.
6. **Safety**: No integrated panic mode, QR address card, or helper‑finding cues.

**AI Buddy differentiators**

-   **Goal‑oriented dialog** with memory (travel tasks as plans).
-   **Multimodal fusion** (vision + maps + speech + user profile).
-   **Food buddy** (dish knowledge, allergen/diet flags, culturally polite orders).
-   **Safety toolkit** (panic translations, QR address card, trusted contacts).
-   **Edge‑first** (low latency, privacy, offline), with **cloud enhancement** for long‑tail.

---

## 6) Feasibility & Technical Options

### 6.1 On‑device viability

-   **ASR**: OpenAI Whisper models range from **tiny (39M)** to **small (244M)** to **medium (769M)**. Quantization (INT8/INT4) can reduce size ~**45%** and cut latency ~**19%** with minimal WER change—enabling mobile inference. Projects like `whisper.cpp` and WhisperKit show real‑time edge feasibility.
-   **OCR**: ML Kit Text Recognition v2 and Apple Vision provide high‑quality on‑device OCR for Latin + major non‑Latin scripts; cloud OCR remains available for complex layouts.
-   **MT**: **NLLB‑200** open‑source model translates **200** languages; distilled/low‑rank variants enable mobile use for common pairs; cloud can handle long‑tail.
-   **TTS**: System TTS (iOS `AVSpeechSynthesizer`, Android engines) for offline; cloud neural TTS (Azure/Google) for premium voices.

### 6.2 Edge vs. Cloud split (policy: **Local‑first, Qualitative fallback**)

-   **Local** by default: OCR, LID, ASR (tiny/base/small quantized), on‑device MT for top pairs, system TTS.
-   **Cloud** when _confidence_ or _coverage_ is low: long‑tail languages, rich reasoning (“explain dish & culture”), premium neural voices.
-   **Confidentiality**: raw audio/video stays on device unless user opts into cloud. When invoking cloud: face/voice redaction if possible + user toggle.

### 6.3 Performance budgets (targets)

-   **Camera AR (OCR→MT)**: P50 < **0.9s**, P95 < **1.2s** per frame batch (1080p).
-   **Speech→Speech**: P50 ≤ **1.8s**, P95 ≤ **2.2s** (5‑sec utterance, quiet/café).
-   **Offline pack sizes**: per‑language core pack ≤ **250MB** (OCR/ASR/MT/TTS combined, tiered by quality).
-   **Battery**: 20‑min continuous use < **12%** on modern devices; thermal step‑down.

### 6.4 System integration

-   **Android**: CameraX, ML Kit OCR, on‑device ASR (whisper.cpp / TFLite), Maps SDK, foreground service for continuous listen, audio focus, Bluetooth mic support.
-   **iOS**: AVFoundation camera, Vision/VisionKit OCR, on‑device speech APIs or WhisperKit, MapKit, background audio modes, Siri Shortcut intents.

---

## 7) Architecture (high level)

```
Camera → OCR (on‑device) ┐
                         ├─ Scene understanding & map context (GPS, POIs)
Mic → VAD → ASR (on‑device) ┘

OCR/ASR text ──► LID & MT (local first; cloud fallback)
                └► Task/goal manager (LLM planner, lightweight on-device where possible)

Planner ► UI/AR overlay (next steps, arrows, pins)
        ► TTS (localized politeness) & phrase cards
        ► Safety module (panic, QR card, share location)
        ► Personalization (dietary profile, saved places)

Model registry & policy: feature flags, confidence gates, telemetry (privacy-preserving)
```

**Vendors/SDKs (build vs. buy matrix)**

-   OCR: **ML Kit**, **Apple Vision**; Cloud: **Google Cloud Vision** when needed.
-   ASR: **Whisper** (quantized) on device; Cloud: Azure/Google STT as fallback.
-   MT: **NLLB‑200** distilled locally; Cloud MT (Google, Azure) for long‑tail.
-   TTS: System TTS local; Cloud: **Azure Neural TTS** / **Google TTS** premium.
-   Maps: **Google Maps SDK / Apple Maps**; offline: **Mapbox**.

---

## 8) Restaurant/Food Intelligence (deep dive)

**Core**

-   Menu OCR + translation; tap dish for **explainer** (ingredients, method, taste profile).
-   **Allergen & diet** mapping (nuts, gluten, shellfish, dairy, pork/alcohol, etc.).
-   **Personalization**: vegetarian/vegan/halal/kosher; highlight safe dishes.
-   **Order assistant**: polite localized phrases (play/display), modifiers ("no onions").

**Data sources & modeling**

-   Curated food ontology + regional variants; few‑shot prompts for LLM to explain dish; risk banners for low confidence; user‑visible sources when online.

**Offline**

-   Pack includes regional dish DB subset; fallback to generic classifications when unknown.

---

## 9) Privacy, Security & Compliance

-   **Default private**: On‑device processing for camera/mic; opt‑in for cloud.
-   **Data minimization**: Only anonymous telemetry (latency/crashes) by default; content never logged unless user opts in.
-   **User control**: Clear toggles for cloud use; per‑item delete; **Erase All**; Face/Touch ID lock for History.
-   **Storage**: Local **AES‑GCM** encryption; TLS 1.3 for any network calls.
-   **Compliance checkpoints**:
    -   **GDPR**: lawful basis, purpose limitation, DPIA for camera/mic contexts, special data handling; video device guidance applies to camera processing contexts.
    -   **CCPA/CPRA**: yearly privacy policy updates; data subject rights (access/delete/correct), opt‑out for “sale/share.”
    -   **App stores**: Apple **Privacy Nutrition Labels**; Android Play privacy policy & data safety labels; upcoming **Accessibility Nutrition Labels**.

> We design for **"local-first"** to simplify compliance and build trust.

---

## 10) Risks & Mitigations

| Risk                         | Impact                    | Mitigation                                                                                   |
| ---------------------------- | ------------------------- | -------------------------------------------------------------------------------------------- |
| Latency on low-end devices   | Poor UX                   | Quantized models; dynamic quality; throttle frame rate; optional cloud                       |
| OCR errors (curvy/low‑light) | Wrong guidance            | Low‑light mode; frame denoise; stabilize boxes; user confirm step                            |
| ASR accents/noise            | Miscomms                  | VAD/denoise, beamforming; diarization; confirmation prompts                                  |
| MT inaccuracies              | Misguidance               | Confidence gating; show originals; one‑tap re-translate via cloud                            |
| Allergen misclassification   | Safety risk               | Conservative classification; explicit disclaimers; show sources; user profile double‑confirm |
| Battery/thermal              | Session drop              | Adaptive performance mode; warn users; pause/resume                                          |
| Regulatory drift             | Store rejection/penalties | Privacy reviews; yearly policy updates; DPIA; regional toggles                               |

---

## 11) Success Metrics (KPIs)

-   **Task success rate** (user confirms “helped”): ≥ **85%**.
-   **Latency**: Camera P50 ≤ **0.9s**; Voice P50 ≤ **1.8s**.
-   **Crash‑free users**: ≥ **99.5%**.
-   **Offline availability**: ≥ **99%** of opens.
-   **D7 retention** (traveler cohort): ≥ **30%**.
-   **Pro conversion** (if monetized): ≥ **8%**.

---

## 12) Research Plan (primary & secondary)

**Primary research**

-   **Diary & intercept studies** in airports, stations, restaurants (Spain, Japan, Thailand; 8–12 participants per city).
-   **Think‑aloud tests** on prototype (lost-in-city + menu flows).
-   **Cafe/street noise** lab tests (65–80 dB) for ASR latency/accuracy.
-   **Accessibility** sessions (VoiceOver/TalkBack users).

**Secondary research**

-   Benchmark against top translators (camera/voice/offline), dedicated devices, and OS features (Live Text; ML Kit). Track language coverage changes quarterly.

**Instrumentation**

-   Privacy‑preserving in‑app surveys; success/failure tagging; opt‑in content/perf logs.

**Sample test matrix**

-   Languages: EN ↔ ES/FR/DE/IT/JA/KO/ZH/AR/PT.
-   Scripts: Latin, CJK, Arabic.
-   Menu types: glossy, handwritten, multi‑column, small font.
-   Lighting: low‑light, glare; motion.

---

## 13) Roadmap (indicative)

**M0 (Tech Spike, 4–6 wks)**: OCR CJK baseline; Whisper quantization; map/AR prototype; allergen ontology draft.  
**M1 (MVP Alpha, ~12 wks)**: Camera OCR→MT overlay; 2‑way voice; offline packs (10 langs); Lost‑in‑city basic; Menu explainer v1.  
**M2 (Beta, ~12–16 wks)**: Safety (panic, QR), helper suggestions, diet profiles, improved AR arrows; more languages (20+).  
**GA (Polish, ~8 wks)**: Accessibility AA, crash‑free ≥99.5%, telemetry, store readme/labels; partnerships.

---

## 14) Monetization (optional)

-   **Free**: Core camera/voice; 2 offline packs; limited history.
-   **Pro ($6–10/mo)**: Unlimited packs, advanced AR, dish explainers, extended history, premium voices.
-   **B2B**: Hotels/airlines/tourism boards SDK/licensing.

---

## 15) References & External Signals

**Apps (features & coverage)**

-   Google Translate (Play Store features incl. instant camera ~94 langs; offline ~59; conversation 70+) — https://play.google.com/store/apps/details?id=com.google.android.apps.translate
-   Google Translate (iOS listing; 249 languages) — https://apps.apple.com/us/app/google-translate/id414706506
-   Google: live conversations & AI updates (Aug 2025) — https://blog.google/products/translate/language-learning-live-translate/
-   Android Circle‑to‑Search “Scroll & translate” (continuous on‑screen translation) — https://www.theverge.com/news/772396/circle-to-search-scroll-translate-continuous-google-android
-   Microsoft Translator features — https://www.microsoft.com/en-us/translator/apps/features/
-   Community report on Android offline changes (2025) — https://answers.microsoft.com/en-us/msoffice/forum/all/translator-on-android-has-changed/cf963888-c277-4efe-80f5-59aaa74a891a
-   DeepL apps — https://www.deepl.com/en/mobile-apps
-   Naver Papago (Play/App Store) — https://play.google.com/store/apps/details?id=com.naver.labs.translator , https://apps.apple.com/us/app/naver-papago-ai-translator/id1147874819
-   iTranslate — https://itranslate.com/ , https://play.google.com/store/apps/details?id=at.nk.tools.iTranslate , https://apps.apple.com/us/app/itranslate-translator/id288113403

**Devices**

-   Timekettle T1 — https://www.timekettle.co/pages/t1 , review — https://www.wired.com/review/timekettle-t1-handheld-translator/
-   Pocketalk device — https://www.pocketalk.com/device , S2 product — https://www.pocketalk.com/product/pocketalk-s2-5-year-esim-white

**Market**

-   UN Tourism Barometer (Jan/May 2025) summary — https://en.unwto-ap.org/news/worldtourismbarometer_jan2025 , https://www.e-unwto.org/doi/abs/10.18111/wtobarometereng.2025.23.1.2
-   WTTC EIR 2024/2025 — https://wttc.org/research/economic-impact
-   Spain 2024 record (AP) — https://apnews.com/article/c9ef4af335ac6194dc4a71f73c4af4eb

**Edge AI feasibility**

-   Whisper model sizes (tiny 39M, small 244M…) — https://huggingface.co/openai/whisper-tiny
-   `whisper.cpp` project — https://github.com/ggml-org/whisper.cpp
-   Quantization effects (size −~45%, latency −~19%) — https://arxiv.org/html/2503.09905v1
-   NLLB‑200 overview — https://ai.meta.com/blog/nllb-200-high-quality-machine-translation/

**OCR & platform**

-   ML Kit Text Recognition v2 — https://developers.google.com/ml-kit/vision/text-recognition/v2/android
-   Apple Vision / VisionKit Live Text — https://developer.apple.com/documentation/vision/recognizing-text-in-images , https://developer.apple.com/documentation/visionkit/enabling-live-text-interactions-with-images
-   Cloud Vision OCR — https://cloud.google.com/vision/docs/ocr

**TTS**

-   Apple AVSpeechSynthesizer — https://developer.apple.com/documentation/avfaudio/avspeechsynthesizer/
-   Azure Neural TTS (140+ languages/variants, 400+ voices) — https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support
-   Google Cloud Text‑to‑Speech voices — https://cloud.google.com/text-to-speech/docs/list-voices-and-types

**Maps**

-   Mapbox pricing/MAU model — https://docs.mapbox.com/android/legacy/maps/guides/pricing/ , https://www.mapbox.com/pricing

**Privacy/Compliance**

-   GDPR video device guidance — https://www.edpb.europa.eu/our-work-tools/our-documents/guidelines/guidelines-32019-processing-personal-data-through-video_en
-   CCPA overview & yearly policy update — https://oag.ca.gov/privacy/ccpa , https://www.workplaceprivacyreport.com/2025/06/articles/california-consumer-privacy-act/ccpa-compliance-reminder-annual-update-requirement-for-online-privacy-policy/
-   Apple Privacy Nutrition Labels — https://www.apple.com/privacy/labels/ , App Store privacy details — https://developer.apple.com/app-store/app-privacy-details/

---

## 16) Next Steps

1. Approve scope & KPIs in §11.
2. Kick off **M0 tech spike** (see §13).
3. Prepare research recruiting for Spain/Japan/Thailand pilots.
4. Draft privacy policy & store labels (templates) aligned to local‑first.

> Appendix pack (on request): benchmark scripts, DPIA template, test matrices, and a sprint‑ready backlog.
