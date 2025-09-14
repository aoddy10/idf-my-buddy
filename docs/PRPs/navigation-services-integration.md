name: "Navigation Services Integration PRP"
description: |

## Purpose

Implement comprehensive GPS-enabled navigation services for the My Buddy AI travel assistant, building upon the completed voice services integration to provide location-aware guidance, turn-by-turn navigation, and contextual points of interest discovery. This phase enables the core travel assistance functionality with voice-guided navigation.

## Core Principles

1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal

Create a complete navigation services integration that provides:

-   GPS location services with privacy-first design
-   Real-time routing and turn-by-turn directions
-   Voice-guided navigation using existing TTS services
-   Points of Interest (POI) discovery and contextual recommendations
-   Multi-modal transportation support (walking, driving, public transit)
-   Integration with voice services for hands-free navigation

## Why

-   **Business value**: Enables core travel assistant navigation functionality
-   **User experience**: Natural voice-guided navigation in multiple languages
-   **Competitive advantage**: Privacy-first location services with edge processing
-   **Integration**: Leverages completed voice services for seamless user interaction
-   **Market need**: Essential feature for travel assistants in unfamiliar locations

## What

### User-Visible Behavior

**Voice-Guided Navigation**

```
User: "Navigate me to the nearest Italian restaurant"
Assistant: "I found 3 Italian restaurants nearby. The closest is Mama Mia's, 0.3 miles away. Shall I start navigation?"
User: "Yes"
Assistant: [Via TTS] "Starting navigation to Mama Mia's. Head north on Main Street for 200 feet, then turn right on Oak Avenue..."
```

**Location Context Awareness**

```
User: "Where am I?"
Assistant: "You're currently on Main Street in downtown San Francisco, near Union Square. There are several restaurants, shops, and the subway station within 2 blocks."
```

**Multi-Modal Route Planning**

```
User: "How do I get to the Golden Gate Bridge?"
Assistant: "I can give you 3 options:
1. Walking: 45 minutes through scenic neighborhoods
2. Driving: 15 minutes via Lombard Street
3. Public transit: 25 minutes via bus line 28
Which would you prefer?"
```

### Technical Requirements

**GPS Integration**

-   Precise location tracking with configurable accuracy
-   Battery-optimized location updates
-   Privacy controls with on-device processing preference
-   Location permission management
-   Offline capability with cached maps

**Routing Services**

-   Integration with Google Maps API and OpenStreetMap
-   Real-time traffic data incorporation
-   Route optimization algorithms
-   Alternative route suggestions
-   Multi-modal transportation options

**Voice Integration**

-   Seamless integration with existing TTS services (pyttsx3, gTTS, etc.)
-   Multilingual navigation instructions using NLLB translation
-   Voice command recognition for navigation controls
-   Audio cues and turn notifications

### Success Criteria

-   [ ] GPS location services working with <10m accuracy
-   [ ] Turn-by-turn navigation with voice guidance in 5+ languages
-   [ ] POI discovery within 1km radius with relevance scoring
-   [ ] Route calculation <3 seconds for local destinations
-   [ ] Voice navigation instructions delivered via existing TTS
-   [ ] Privacy controls allowing full offline mode
-   [ ] Integration tests passing with 90%+ coverage
-   [ ] Performance benchmark: <1s response for "Where am I?"
-   [ ] Cross-platform compatibility (iOS/Android/Web)

## All Needed Context

### Documentation & References

```yaml
# MUST READ - Include these in your context window

- url: https://developers.google.com/maps/documentation/directions/overview
  why: Official Google Maps Directions API - routing algorithms and parameters
  critical: Rate limiting, API keys, response format

- url: https://developers.google.com/maps/documentation/places/web-service/overview
  why: Places API for POI discovery and contextual location data
  critical: Place types, search radius, result ranking

- url: https://wiki.openstreetmap.org/wiki/API
  why: OpenStreetMap API as backup/privacy alternative to Google Maps
  critical: Licensing, data format, offline capabilities

- file: app/services/whisper.py
  why: Pattern for AI service initialization with model management
  critical: Device selection, caching, error handling patterns

- file: app/services/nllb.py
  why: Translation service integration for multilingual navigation
  critical: Language detection, fallback mechanisms, async patterns

- file: app/api/v1/voice/conversation.py
  why: WebSocket integration pattern for real-time features
  critical: Connection management, error handling, streaming

- file: app/core/config.py
  why: Configuration management for API keys and service settings
  critical: Environment variables, secrets management

- doc: https://geoalchemy-2.readthedocs.io/en/latest/
  section: Spatial data types and queries
  critical: GPS coordinate storage and spatial indexing

- docfile: MULTILINGUAL_VERIFICATION_REPORT.md
  why: Multilingual support patterns and validation approaches
  critical: Language support matrix, testing methodology
```

### Current Codebase Structure

```
app/
├── api/
│   ├── v1/
│   │   ├── navigation.py         # ← Main implementation target
│   │   └── voice/
│   │       └── conversation.py   # ← Integration point for voice nav
├── services/
│   ├── whisper.py               # ← Voice recognition integration
│   ├── nllb.py                  # ← Translation for multilingual nav
│   ├── tts.py                   # ← Voice guidance output
│   └── maps.py                  # ← NEW: Core navigation service
├── models/
│   └── entities/
│       └── location.py          # ← NEW: GPS/location data models
├── schemas/
│   └── navigation.py            # ← NEW: API request/response schemas
└── core/
    ├── config.py                # ← API key configuration
    └── deps.py                  # ← Service dependency injection
```

## Technical Implementation Strategy

### Phase 1: Core GPS Services (Week 1)

**Service Implementation**

```python
# app/services/maps.py - Core navigation service
class NavigationService:
    def __init__(self):
        self.google_maps_client = None
        self.osm_client = None
        self._setup_clients()

    async def get_current_location(self) -> LocationInfo
    async def calculate_route(self, origin, destination, mode) -> RouteInfo
    async def get_nearby_pois(self, location, radius, types) -> List[POI]
```

**Data Models**

```python
# app/models/entities/location.py
class Location(SQLModel):
    latitude: float
    longitude: float
    accuracy: Optional[float]
    timestamp: datetime

class Route(SQLModel):
    origin: Location
    destination: Location
    waypoints: List[Location]
    distance_meters: int
    duration_seconds: int
    instructions: List[str]
```

### Phase 2: Voice-Guided Navigation (Week 2)

**Voice Integration**

```python
# Integration with existing TTS services
async def speak_navigation_instruction(
    instruction: str,
    language: str = "en"
) -> None:
    # Use existing TTS services from voice integration
    translated = await translation_service.translate_text(
        instruction, target_language=language
    )
    await tts_service.speak(translated["translated_text"])
```

**API Endpoints**

```python
# app/api/v1/navigation.py
@router.post("/navigate/start")
async def start_navigation(request: NavigationRequest)

@router.get("/location/current")
async def get_current_location()

@router.get("/poi/nearby")
async def get_nearby_points_of_interest()
```

### Phase 3: Advanced Features (Week 3)

**Real-time Updates**

-   WebSocket integration for live navigation updates
-   Traffic-aware route recalculation
-   POI recommendations based on user preferences

**Privacy & Offline**

-   Cached map data for offline navigation
-   On-device route calculation options
-   Privacy mode with minimal data sharing

## Validation & Testing Strategy

### Progressive Validation Checklist

```markdown
## Phase 1 Validation (GPS Services)

-   [ ] GPS service initializes without errors
-   [ ] Location retrieval works with mock coordinates
-   [ ] Route calculation returns valid paths
-   [ ] POI discovery finds nearby locations
-   [ ] Error handling covers API failures
-   [ ] Configuration loads API keys correctly

## Phase 2 Validation (Voice Integration)

-   [ ] Navigation instructions speak via TTS
-   [ ] Multilingual support works (test EN, ES, FR)
-   [ ] Voice commands trigger navigation actions
-   [ ] WebSocket navigation updates work
-   [ ] Audio cues play at appropriate times
-   [ ] Integration with existing voice pipeline

## Phase 3 Validation (Advanced Features)

-   [ ] Real-time traffic updates incorporated
-   [ ] Offline mode works with cached data
-   [ ] Privacy controls function properly
-   [ ] Performance meets <3s route calculation
-   [ ] Cross-platform compatibility verified
-   [ ] Full integration test suite passes
```

### Test Implementation

**Unit Tests**

```python
# tests/services/test_navigation.py
class TestNavigationService:
    def test_gps_location_parsing(self):
    def test_route_calculation_valid(self):
    def test_poi_discovery_filters(self):
    def test_voice_integration(self):
```

**Integration Tests**

```python
# tests/integration/test_voice_navigation.py
class TestVoiceNavigation:
    def test_start_navigation_voice_flow(self):
    def test_multilingual_navigation_instructions(self):
    def test_realtime_navigation_updates(self):
```

**Performance Tests**

```python
# tests/performance/test_navigation_performance.py
def test_route_calculation_speed():
    # Target: <3s for local routes

def test_location_accuracy():
    # Target: <10m accuracy

def test_voice_instruction_latency():
    # Target: <1s from turn to audio output
```

## Dependencies & Prerequisites

### External Dependencies

```toml
# Add to pyproject.toml
googlemaps = "^4.10.0"          # Google Maps integration
overpass = "^0.7"               # OpenStreetMap queries
geopy = "^2.4.0"                # Geographic calculations
geoalchemy2 = "^0.14.0"         # Spatial database support
shapely = "^2.0.0"              # Geometric operations
```

### Service Dependencies

-   ✅ Voice Services (Whisper ASR, TTS, NLLB translation) - COMPLETED
-   ✅ Core FastAPI infrastructure - COMPLETED
-   ✅ Database with SQLModel - COMPLETED
-   📋 Google Maps API key configuration - NEEDED
-   📋 OpenStreetMap integration setup - NEEDED

### Configuration Updates

```python
# app/core/config.py additions
class Settings(BaseSettings):
    # Existing settings...

    # Navigation service settings
    google_maps_api_key: str = Field(..., env="GOOGLE_MAPS_API_KEY")
    enable_offline_maps: bool = Field(True, env="ENABLE_OFFLINE_MAPS")
    location_accuracy_meters: int = Field(10, env="LOCATION_ACCURACY_METERS")
    poi_search_radius_km: float = Field(1.0, env="POI_SEARCH_RADIUS_KM")
```

## Success Metrics & Benchmarks

### Performance Targets

-   **Location Accuracy**: <10 meters in urban areas
-   **Route Calculation**: <3 seconds for local destinations (<50km)
-   **POI Discovery**: <2 seconds for nearby search (1km radius)
-   **Voice Instruction Latency**: <1 second from navigation event to audio
-   **Battery Impact**: <5% additional drain during active navigation

### Quality Targets

-   **API Uptime**: 99.5% availability during development
-   **Translation Accuracy**: >90% for navigation instructions
-   **Voice Recognition**: >85% accuracy for navigation commands
-   **Cross-Platform**: Works on iOS, Android, and Web equally
-   **Offline Capability**: Basic navigation works without internet

## Risk Mitigation

### Technical Risks

-   **API Rate Limits**: Implement caching and OpenStreetMap fallback
-   **GPS Accuracy**: Multiple positioning methods (GPS, WiFi, Cell tower)
-   **Battery Drain**: Configurable update intervals and background processing
-   **Privacy Concerns**: On-device processing options and clear data controls

### Integration Risks

-   **Voice Service Conflicts**: Careful audio session management
-   **Performance Impact**: Background processing optimization
-   **Platform Differences**: Abstracted location services with platform adapters

## Definition of Done

This PRP is complete when:

1. **Core Services**: GPS location, routing, and POI discovery work reliably
2. **Voice Integration**: Navigation instructions delivered via existing TTS in 5+ languages
3. **API Coverage**: All planned endpoints implemented with proper error handling
4. **Testing**: 90%+ test coverage with unit, integration, and performance tests
5. **Documentation**: API documentation and user guide complete
6. **Performance**: All benchmarks met consistently
7. **Privacy**: Offline mode and privacy controls functional
8. **Integration**: Seamless connection with existing voice services

**Validation Command**: `python -m pytest tests/integration/test_navigation.py -v --cov=app/services/maps --cov=app/api/v1/navigation`

---

## Implementation Notes

### Learning from Voice Services Success

Based on the successful completion of Phase 1 (Voice Services Integration), apply these proven patterns:

1. **Progressive Implementation**: Start with basic GPS → Add routing → Add voice integration
2. **Comprehensive Testing**: Create validation checklist similar to voice services (9 items)
3. **Multi-Backend Strategy**: Google Maps + OpenStreetMap (like NLLB + Google Translate)
4. **Performance Focus**: Benchmark and optimize for <3s response times
5. **Error Handling**: Graceful degradation and fallback mechanisms
6. **Service Architecture**: Follow the pattern established in whisper.py and nllb.py

### Next Phase Preparation

This navigation integration prepares the foundation for:

-   **Phase 3**: Restaurant Intelligence (location-aware restaurant discovery)
-   **Phase 4**: Shopping Assistant (location-based shopping recommendations)
-   **Phase 5**: Safety & Emergency (location-aware emergency services)

The robust navigation services will enable location context for all subsequent travel assistant features.
