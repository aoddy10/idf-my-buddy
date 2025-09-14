name: "Restaurant Intelligence PRP"
description: |

## Purpose

Implement comprehensive restaurant intelligence services for the My Buddy AI travel assistant, building upon completed voice services and ready navigation services to provide intelligent menu translation, allergen detection, dish explanations, and personalized restaurant recommendations. This phase enables users to navigate foreign dining experiences confidently with AI-powered culinary assistance.

## Core Principles

1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal

Create a complete restaurant intelligence system that provides:

-   High-accuracy menu OCR with multi-language support
-   Real-time dish translation and cultural explanations
-   Comprehensive allergen detection and dietary restriction matching
-   Personalized restaurant recommendations using location context
-   Voice-guided dining assistance integrated with existing TTS services
-   Cultural dining etiquette and local customs guidance

## Why

-   **Business value**: Enables confident dining experiences in foreign locations
-   **User safety**: Critical allergen detection prevents health emergencies
-   **Cultural bridge**: Explains unfamiliar dishes and dining customs
-   **Integration foundation**: Leverages completed voice + navigation services
-   **Market differentiation**: AI-powered culinary intelligence for travelers

## What

### User-Visible Behavior

**Menu Translation & Explanation**

```
User: [Takes photo of Thai menu]
Assistant: [Via TTS] "I can see this Thai menu. Let me translate the dishes for you. The first item is 'Pad Thai' - stir-fried rice noodles with shrimp, tofu, and tamarind sauce. It's Thailand's most famous dish, mildly sweet and tangy. Would you like me to check it against your allergies?"

User: "Yes, check for shellfish"
Assistant: "Warning: This dish contains shrimp, which is a shellfish. I found 3 shellfish-free alternatives on this menu. Would you like recommendations?"
```

**Allergen Detection & Safety**

```
User: [Shows menu item] "Is this safe for my nut allergy?"
Assistant: [Analyzing menu text] "I can see this dish contains peanuts in the sauce. This is NOT safe for your nut allergy. Let me find similar dishes without nuts... I found 'Pad See Ew' which is similar noodles but nut-free."
```

**Restaurant Discovery & Context**

```
User: "Find me authentic local food nearby"
Assistant: "I found 3 highly-rated local restaurants within walking distance:
1. Som Tam Nua - Famous for papaya salad, very authentic, spicy food
2. Krua Apsorn - Royal Thai cuisine, milder flavors, tourist-friendly
3. Jay Fai - Michelin starred street food, long wait but incredible

Based on your mild spice preference, I recommend Krua Apsorn. Shall I start navigation?"
```

**Cultural Dining Guidance**

```
User: "How do I order here? What's polite?"
Assistant: "In Thai restaurants, it's polite to share dishes family-style. Order 1-2 dishes per person for the table. Use the serving spoon, not your personal spoon for shared dishes. Tipping is not expected but 10% is appreciated in tourist areas."
```

### Technical Requirements

**Menu OCR Enhancement**

-   Multi-language OCR with preprocessing for restaurant menus
-   Handle varied lighting, angles, and menu formats
-   Extract structured data: dish names, descriptions, prices
-   Confidence scoring for text extraction accuracy

**Intelligent Translation & Explanation**

-   Integration with NLLB for dish name translation
-   Cultural context and cooking method explanations
-   Ingredient identification and preparation descriptions
-   Local/regional dish variations and authenticity notes

**Allergen Detection System**

-   Comprehensive allergen database with regional variations
-   Cross-contamination risk assessment
-   Ingredient analysis using ML classification
-   User allergy profile matching and warnings

**Restaurant Intelligence**

-   Location-aware restaurant discovery using navigation services
-   Rating aggregation from multiple sources
-   Cultural authenticity scoring
-   Personalized recommendations based on dietary needs

### Success Criteria

-   [ ] OCR accuracy >90% for restaurant menus in 10+ languages
-   [ ] Allergen detection with >99% recall for major allergens
-   [ ] Dish translation and explanation in <3 seconds
-   [ ] Restaurant recommendations with location integration
-   [ ] Voice-guided menu assistance using existing TTS
-   [ ] Cultural dining guidance database with 50+ countries
-   [ ] Integration tests passing with 95%+ coverage
-   [ ] Performance: <2s menu scan to recommendations
-   [ ] Safety: Zero false negatives for critical allergen detection
-   [ ] User experience: Natural conversation flow with voice services

## All Needed Context

### Documentation & References

```yaml
# MUST READ - Include these in your context window

- url: https://github.com/JaidedAI/EasyOCR
  why: Primary OCR engine - multi-language text extraction from images
  critical: Language support, model loading, preprocessing requirements

- url: https://tesseract-ocr.github.io/
  why: Fallback OCR engine - mature text recognition with preprocessing
  critical: Language data files, config options, accuracy optimization

- url: https://opencv.org/
  why: Image preprocessing - noise reduction, perspective correction
  critical: Image enhancement techniques, format handling

- url: https://ai.meta.com/research/no-language-left-behind/
  why: Translation integration - already implemented in voice services
  critical: Language pair coverage, batch processing, performance

- url: https://www.food.gov.uk/business-guidance/allergen-guidance-for-food-businesses
  why: Allergen regulation standards and detection requirements
  critical: Cross-contamination rules, labeling standards, safety protocols

- url: https://developers.google.com/maps/documentation/places/web-service/overview
  why: Restaurant discovery using navigation services integration
  critical: Place types, review aggregation, location-based search

- file: app/services/nllb.py
  why: Translation service pattern - async implementation with fallbacks
  critical: Error handling, performance optimization, model management

- file: app/services/ocr.py
  why: Existing OCR infrastructure - multi-backend support pattern
  critical: Backend selection logic, preprocessing pipeline, error handling

- file: app/schemas/restaurant.py
  why: Data models for restaurants, menus, allergens, dietary restrictions
  critical: Schema validation, enum definitions, relationship modeling

- file: app/api/v1/voice/conversation.py
  why: Voice integration pattern for restaurant assistance
  critical: WebSocket integration, conversation state, TTS coordination

- file: app/core/config.py
  why: Configuration management for OCR models and API keys
  critical: Model paths, service enablement flags, performance settings

- docfile: MULTILINGUAL_VERIFICATION_REPORT.md
  why: Multilingual support validation methodology
  critical: Language testing matrix, accuracy measurement, edge cases
```

### Current Codebase Structure

```
app/
├── api/
│   ├── restaurant.py             # ← MAIN TARGET - implement full API
│   └── v1/voice/
│       └── conversation.py       # ← Integration point for voice assistance
├── services/
│   ├── ocr.py                   # ← ENHANCE - menu-specific processing
│   ├── nllb.py                  # ← INTEGRATE - dish translation
│   ├── tts.py                   # ← INTEGRATE - voice explanations
│   └── restaurant_intelligence.py # ← NEW - core intelligence service
├── models/entities/
│   ├── restaurant.py            # ← NEW - restaurant/menu data models
│   └── user.py                  # ← ENHANCE - dietary preferences
├── schemas/
│   ├── restaurant.py            # ← ENHANCE - comprehensive API schemas
│   └── common.py                # ← ENHANCE - allergen/dietary enums
└── utils/
    ├── menu_parser.py           # ← NEW - structured menu extraction
    ├── allergen_detector.py     # ← NEW - safety analysis system
    └── culture_guide.py         # ← NEW - dining etiquette database
```

### Desired Codebase Structure After Implementation

```
app/
├── services/
│   ├── restaurant_intelligence.py  # Core service orchestrating all features
│   ├── menu_ocr.py                 # Enhanced OCR for menu processing
│   ├── allergen_detection.py       # Safety-critical allergen analysis
│   ├── dish_explainer.py           # Cultural context and descriptions
│   └── restaurant_discovery.py     # Location-aware recommendations
├── models/entities/
│   ├── menu_item.py               # Structured menu data
│   ├── allergen_profile.py        # User allergy/dietary data
│   └── restaurant_review.py       # Aggregated rating data
├── utils/
│   ├── image_preprocessor.py      # Menu photo enhancement
│   ├── cultural_database.py       # Dining customs and etiquette
│   └── nutrition_analyzer.py      # Dish nutrition estimation
└── tests/
    ├── test_menu_ocr.py           # OCR accuracy testing
    ├── test_allergen_detection.py  # Safety validation testing
    └── integration/
        └── test_restaurant_flow.py # End-to-end user workflows
```

### Known Gotchas & Library Quirks

```python
# CRITICAL: EasyOCR model loading patterns
# Model downloads ~100MB on first use per language
# Must handle offline scenarios gracefully
reader = easyocr.Reader(['en', 'th'], gpu=False)  # CPU for edge devices

# CRITICAL: OpenCV image preprocessing requirements
# Menu photos need perspective correction and noise reduction
# Must handle various lighting conditions and angles
img = cv2.bilateralFilter(img, 9, 75, 75)  # Preserve edges while reducing noise

# CRITICAL: Allergen detection safety requirements
# FALSE NEGATIVES are dangerous - err on side of caution
# Must handle ingredient synonyms and cross-contamination
ALLERGEN_KEYWORDS = {
    'peanuts': ['peanut', 'groundnut', 'arachis oil', 'mandalona nuts']
}

# CRITICAL: NLLB language detection for menus
# Menu text often mixed languages (English + local)
# Need preprocessing to separate language segments
detected_lang = detect_language_segments(menu_text)

# CRITICAL: Performance for real-time menu scanning
# OCR + Translation + Analysis must complete <3 seconds
# Use async processing and smart caching
async with asyncio.TaskGroup() as tg:
    ocr_task = tg.create_task(extract_menu_text(image))
    translation_task = tg.create_task(translate_batch(text_segments))
```

## Implementation Blueprint

### Data Models and Structure

Create comprehensive data models ensuring type safety and relationship integrity:

```python
# Core entities for restaurant intelligence
class MenuItem(SQLModel, table=True):
    item_id: str = Field(primary_key=True)
    name: str
    translated_name: Optional[str]
    description: Optional[str]
    translated_description: Optional[str]
    price: Optional[Decimal]
    currency: str = "USD"
    allergens: List[AllergenType]
    dietary_flags: List[DietaryRestriction]
    spice_level: Optional[int] = Field(ge=0, le=5)
    cultural_notes: Optional[str]

class AllergenProfile(SQLModel, table=True):
    user_id: str = Field(foreign_key="user.id")
    allergens: List[AllergenType]
    severity_levels: Dict[str, str]  # mild/moderate/severe
    cross_contamination_sensitive: bool = False

class MenuScan(SQLModel, table=True):
    scan_id: str = Field(primary_key=True)
    image_path: str
    extracted_text: str
    confidence_score: float
    detected_language: str
    processed_items: List[MenuItem]
    scan_timestamp: datetime
```

### Task List for Implementation

```yaml
Task 1: Enhance OCR Service for Menu Processing
MODIFY app/services/ocr.py:
    - ADD menu-specific preprocessing pipeline
    - IMPLEMENT multi-language detection for mixed content
    - ADD confidence scoring and quality assessment
    - CREATE structured text extraction for menu formats

Task 2: Create Restaurant Intelligence Service
CREATE app/services/restaurant_intelligence.py:
    - IMPLEMENT MenuIntelligenceService class
    - ADD integration with OCR, NLLB, and navigation services
    - CREATE dish explanation and cultural context system
    - ADD performance monitoring and caching

Task 3: Implement Allergen Detection System
CREATE app/services/allergen_detection.py:
    - BUILD comprehensive allergen keyword database
    - IMPLEMENT ingredient analysis with ML classification
    - ADD cross-contamination risk assessment
    - CREATE safety-critical validation with zero false negatives

Task 4: Create Dish Explanation Service
CREATE app/services/dish_explainer.py:
    - INTEGRATE NLLB for dish name translation
    - ADD cultural context database for regional cuisines
    - IMPLEMENT cooking method and ingredient explanations
    - CREATE authenticity scoring for tourist recommendations

Task 5: Enhance Restaurant API Endpoints
MODIFY app/api/restaurant.py:
    - IMPLEMENT /scan-menu endpoint for photo processing
    - ADD /explain-dish endpoint with voice integration
    - CREATE /check-allergens safety endpoint
    - ADD /recommend-restaurants with location integration

Task 6: Create Menu Data Models
CREATE app/models/entities/menu_item.py:
    - IMPLEMENT MenuItem, MenuScan, AllergenProfile models
    - ADD relationship definitions with User and Restaurant
    - CREATE database migration for new tables
    - ADD indexes for performance optimization

Task 7: Enhance Restaurant Schemas
MODIFY app/schemas/restaurant.py:
    - ADD MenuScanRequest/Response with image handling
    - IMPLEMENT DishExplanationRequest with cultural context
    - CREATE AllergenCheckRequest with safety warnings
    - ADD RestaurantRecommendationResponse with reasoning

Task 8: Create Image Processing Utilities
CREATE app/utils/image_preprocessor.py:
    - IMPLEMENT perspective correction for menu photos
    - ADD noise reduction and enhancement filters
    - CREATE automatic rotation and cropping
    - ADD format normalization pipeline

Task 9: Build Cultural Dining Database
CREATE app/utils/cultural_database.py:
    - COMPILE dining etiquette by country/region
    - ADD typical dish characteristics and ingredients
    - IMPLEMENT cultural context for unfamiliar foods
    - CREATE tipping and payment custom guidelines

Task 10: Integrate Voice-Guided Restaurant Assistance
MODIFY app/api/v1/voice/conversation.py:
    - ADD restaurant assistance conversation flows
    - INTEGRATE TTS for menu explanations and warnings
    - CREATE voice commands for allergen checking
    - ADD cultural dining guidance via voice
```

### Per-Task Pseudocode

```python
# Task 1: Enhanced Menu OCR
async def process_menu_image(image: bytes) -> MenuScanResult:
    # PATTERN: Multi-stage processing pipeline (see whisper.py)
    preprocessed = await enhance_menu_image(image)

    # GOTCHA: EasyOCR requires specific image format
    pil_image = Image.open(io.BytesIO(preprocessed))

    # CRITICAL: Handle mixed languages in menu text
    text_segments = await self._ocr_reader.readtext(
        np.array(pil_image),
        detail=1  # Returns bounding boxes + confidence
    )

    # PATTERN: Confidence-based quality assessment
    high_confidence_text = [
        segment for segment in text_segments
        if segment[2] > settings.ocr.min_confidence
    ]

    return MenuScanResult(
        extracted_text=combine_segments(high_confidence_text),
        confidence_score=calculate_average_confidence(text_segments),
        detected_language=detect_primary_language(high_confidence_text)
    )

# Task 3: Safety-Critical Allergen Detection
async def check_allergens(
    menu_item: MenuItem,
    user_allergies: List[AllergenType]
) -> AllergenCheckResult:

    # CRITICAL: Zero false negatives for safety
    detected_allergens = []

    for allergy in user_allergies:
        # Check direct matches and synonyms
        if await self._contains_allergen(menu_item.description, allergy):
            detected_allergens.append(allergy)

    # PATTERN: Conservative risk assessment
    risk_level = "HIGH" if detected_allergens else "LOW"

    # GOTCHA: Must check cross-contamination warnings
    if user_profile.cross_contamination_sensitive:
        cross_risk = await self._assess_cross_contamination(menu_item)
        if cross_risk:
            risk_level = "MEDIUM"

    return AllergenCheckResult(
        safe_to_consume=len(detected_allergens) == 0,
        detected_allergens=detected_allergens,
        risk_level=risk_level,
        alternative_suggestions=await self._suggest_alternatives(menu_item)
    )

# Task 5: Voice-Integrated Restaurant API
@router.post("/explain-dish")
async def explain_dish_with_voice(
    request: DishExplanationRequest
) -> DishExplanationResponse:

    # PATTERN: Integration with existing services
    explanation = await restaurant_service.explain_dish(
        dish_name=request.dish_name,
        target_language=request.language
    )

    # PATTERN: Voice integration (see conversation.py)
    if request.include_voice_explanation:
        audio_explanation = await tts_service.synthesize_speech(
            text=explanation.cultural_context,
            language=request.language,
            voice_settings=VoiceSettings(speed=0.9)  # Slower for learning
        )
        explanation.audio_data = audio_explanation

    return DishExplanationResponse(
        dish_explanation=explanation,
        cultural_context=explanation.cultural_context,
        ingredients_breakdown=explanation.ingredients,
        authenticity_score=explanation.authenticity_rating
    )
```

### Integration Points

```yaml
VOICE SERVICES:
    - integration: Use existing TTS for dish explanations
    - pattern: "await tts_service.speak(explanation, language=user_lang)"
    - critical: Handle long explanations with natural pauses

NAVIGATION SERVICES:
    - integration: Restaurant discovery using location context
    - pattern: "nearby_restaurants = await navigation_service.search_places(coords, 'restaurant')"
    - critical: Distance-based filtering and walking time estimation

DATABASE:
    - migration: "Add allergen_profiles, menu_scans, cultural_notes tables"
    - indexes: "CREATE INDEX idx_allergen_lookup ON menu_items(allergens)"
    - critical: Foreign key relationships with user preferences

CONFIG:
    - add to: app/core/config.py
    - pattern: "OCR_CONFIDENCE_THRESHOLD = float(os.getenv('OCR_CONFIDENCE', '0.8'))"
    - critical: Model paths, language packs, safety thresholds

ROUTES:
    - add to: app/main.py
    - pattern: "app.include_router(restaurant_router, prefix='/restaurant')"
    - critical: Error handling for image upload size limits
```

## Validation Loop

### Level 1: Syntax & Style

```bash
# CRITICAL: Run these FIRST - fix any errors before proceeding
ruff check app/services/restaurant_intelligence.py --fix
ruff check app/api/restaurant.py --fix
mypy app/services/ --strict
mypy app/api/ --strict

# Expected: No errors. If errors exist, READ carefully and fix.
```

### Level 2: Unit Tests - Safety Critical Components

```python
# tests/services/test_allergen_detection.py - SAFETY CRITICAL
def test_allergen_detection_no_false_negatives():
    """CRITICAL: Must detect all allergens - no false negatives allowed."""
    test_cases = [
        ("Contains peanuts and tree nuts", ["peanuts", "tree_nuts"]),
        ("Made with arachis oil", ["peanuts"]),  # Synonym test
        ("May contain traces of shellfish", ["shellfish"]),  # Cross-contamination
    ]

    for description, expected_allergens in test_cases:
        detected = allergen_detector.detect_allergens(description)
        for allergen in expected_allergens:
            assert allergen in detected, f"SAFETY FAILURE: {allergen} not detected in '{description}'"

def test_ocr_menu_accuracy():
    """Test OCR accuracy on sample menu images."""
    sample_menus = load_test_menu_images()

    for menu_image, expected_text in sample_menus:
        result = await ocr_service.process_menu_image(menu_image)
        accuracy = calculate_text_similarity(result.extracted_text, expected_text)
        assert accuracy > 0.85, f"OCR accuracy {accuracy} below threshold"

def test_dish_translation_quality():
    """Test dish name translation accuracy."""
    test_dishes = [
        ("Pad Thai", "th", "Stir-fried rice noodles"),
        ("Coq au Vin", "fr", "Chicken braised in wine"),
        ("Ratatouille", "fr", "Vegetable stew")
    ]

    for dish_name, source_lang, expected_concept in test_dishes:
        translation = await dish_explainer.explain_dish(dish_name, source_lang)
        assert expected_concept.lower() in translation.explanation.lower()
```

```bash
# Run safety-critical tests first
uv run pytest tests/services/test_allergen_detection.py -v --tb=short
# MUST PASS: Zero tolerance for allergen detection failures

uv run pytest tests/services/test_restaurant_intelligence.py -v
# If failing: Check error logs, understand root cause, fix code
```

### Level 3: Integration Test with Voice Services

```bash
# Start the service
uv run python -m app.main --dev

# Test menu scanning endpoint
curl -X POST http://localhost:8000/restaurant/scan-menu \
  -F "image=@test_menu.jpg" \
  -F "language=en" \
  -F "check_allergens=true" \
  -F "user_allergies=[\"peanuts\",\"shellfish\"]"

# Expected: JSON response with extracted menu items, translations, allergen warnings
# If error: Check logs at logs/app.log for OCR/translation failures

# Test voice-guided dish explanation
curl -X POST http://localhost:8000/restaurant/explain-dish \
  -H "Content-Type: application/json" \
  -d '{"dish_name": "Pad Thai", "language": "en", "include_voice": true}'

# Expected: Dish explanation with cultural context + audio file
# If error: Check TTS integration and NLLB translation logs
```

### Level 4: Performance & Safety Validation

```python
# Performance benchmarks
def test_menu_processing_speed():
    """Menu scan to recommendation must complete <3 seconds."""
    start_time = time.time()

    result = await restaurant_service.process_menu_workflow(
        image=sample_menu_image,
        user_preferences=test_user_profile
    )

    processing_time = time.time() - start_time
    assert processing_time < 3.0, f"Processing took {processing_time}s, exceeds 3s limit"

def test_allergen_safety_comprehensive():
    """Comprehensive allergen safety testing with edge cases."""
    dangerous_test_cases = [
        "Contains nuts (may include peanuts)",  # Ambiguous wording
        "Processed in facility with shellfish",  # Cross-contamination
        "Arachis hypogaea oil",  # Scientific name for peanuts
        "Tree nuts: almonds, walnuts",  # Multiple allergens
    ]

    user_with_severe_allergies = AllergenProfile(
        allergens=[AllergenType.PEANUTS, AllergenType.TREE_NUTS],
        cross_contamination_sensitive=True
    )

    for test_description in dangerous_test_cases:
        safety_result = await allergen_detector.assess_safety(
            test_description, user_with_severe_allergies
        )
        assert not safety_result.safe_to_consume, f"SAFETY FAILURE: Incorrectly marked safe: {test_description}"
```

## Final Validation Checklist

-   [ ] All tests pass: `uv run pytest tests/ -v --cov=app/services --cov=app/api`
-   [ ] No linting errors: `uv run ruff check app/`
-   [ ] No type errors: `uv run mypy app/ --strict`
-   [ ] OCR accuracy >90%: Manual testing with 10 diverse menu images
-   [ ] Allergen detection 100% recall: Zero false negatives in safety tests
-   [ ] Menu processing <3s: Performance benchmark with realistic images
-   [ ] Voice integration works: TTS explanations play correctly
-   [ ] Cultural context accurate: Manual review of dish explanations
-   [ ] API documentation complete: OpenAPI schema generated correctly

## Anti-Patterns to Avoid

-   ❌ Don't skip allergen detection validation - safety is critical
-   ❌ Don't hardcode allergen keywords - use configurable database
-   ❌ Don't ignore OCR confidence scores - filter low-quality extractions
-   ❌ Don't block UI on slow OCR - use async processing with progress updates
-   ❌ Don't store user photos permanently - privacy-first processing
-   ❌ Don't trust single OCR backend - implement fallback mechanisms
-   ❌ Don't oversimplify cultural explanations - provide genuine context
-   ❌ Don't ignore cross-contamination risks - conservative safety approach

---

## Success Metrics Target

**Confidence Level for One-Pass Implementation: 8.5/10**

**Reasoning:**

-   **Strengths (8.5)**: Comprehensive context provided, existing OCR/translation infrastructure, well-defined schemas, clear safety requirements, detailed validation approach
-   **Complexity factors (-1.5)**: Multi-service integration complexity, safety-critical allergen detection requirements, cultural database creation scope

**Risk Mitigation:**

-   Start with basic OCR → Add translation → Add safety features → Add cultural intelligence
-   Implement comprehensive test suite with safety validation at each step
-   Use existing patterns from voice/navigation services for consistency
-   Focus on progressive enhancement with validation loops
