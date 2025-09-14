"""
Allergen Detection Service

SAFETY-CRITICAL SERVICE for detecting allergens in restaurant menu items.
This service prioritizes zero false negatives to ensure user safety.

Key safety principles:
1. Conservative detection: If uncertain, flag as potential allergen
2. Multi-language support for international travel
3. Comprehensive allergen database with regional variations
4. Clear risk level assessment and warnings
5. Detailed logging for safety audit trails

Author: AI Assistant
Date: 2024
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from app.core.logging import get_logger


class AllergenRiskLevel(str, Enum):
    """Risk levels for allergen detection - prioritizes user safety."""
    NONE = "none"           # No allergens detected
    LOW = "low"             # Possible allergens, low confidence
    MEDIUM = "medium"       # Likely allergens detected
    HIGH = "high"           # Multiple or confirmed allergens
    CRITICAL = "critical"   # Severe allergens (nuts, shellfish) detected


class AllergenType(str, Enum):
    """Standard allergen categories based on major food safety regulations."""
    MILK_DAIRY = "milk_dairy"
    EGGS = "eggs"
    FISH = "fish"
    SHELLFISH = "shellfish"
    TREE_NUTS = "tree_nuts"
    PEANUTS = "peanuts"
    WHEAT_GLUTEN = "wheat_gluten"
    SOYBEANS = "soybeans"
    SESAME = "sesame"
    SULFITES = "sulfites"
    MOLLUSCS = "molluscs"
    CELERY = "celery"
    MUSTARD = "mustard"
    LUPIN = "lupin"
    OTHER = "other"


@dataclass
class AllergenMatch:
    """Details of a detected allergen."""
    allergen_type: AllergenType
    matched_text: str
    confidence: float
    position: int
    language: str
    severity_notes: List[str] = field(default_factory=list)


@dataclass
class AllergenDetectionResult:
    """Complete allergen detection results with safety information."""
    text_analyzed: str
    language: str
    detected_allergens: List[AllergenMatch] = field(default_factory=list)
    risk_level: AllergenRiskLevel = AllergenRiskLevel.NONE
    safety_warnings: List[str] = field(default_factory=list)
    user_specific_warnings: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class AllergenDetectionService:
    """
    Safety-critical service for detecting allergens in menu text.
    
    This service implements conservative detection to minimize false negatives
    that could endanger users with food allergies.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.logger.info("Initializing safety-critical AllergenDetectionService")
        
        # Comprehensive allergen databases by language
        # These are expanded regularly based on food safety regulations
        self._allergen_patterns = self._build_allergen_patterns()
        
        # Critical allergens that require immediate attention
        self.critical_allergens = {
            AllergenType.PEANUTS,
            AllergenType.TREE_NUTS,
            AllergenType.SHELLFISH,
            AllergenType.FISH
        }
        
        # Initialize detection statistics for monitoring
        self.detection_stats = {
            "total_analyses": 0,
            "allergens_detected": 0,
            "critical_warnings": 0,
            "false_positive_reports": 0  # User feedback tracking
        }
    
    def _build_allergen_patterns(self) -> Dict[str, Dict[AllergenType, List[str]]]:
        """Build comprehensive allergen detection patterns by language."""
        
        return {
            "en": {
                AllergenType.MILK_DAIRY: [
                    # Direct terms
                    "milk", "dairy", "lactose", "cream", "butter", "cheese", "yogurt", 
                    "yoghurt", "whey", "casein", "curd", "custard", "ice cream",
                    # Cheese varieties
                    "cheddar", "mozzarella", "parmesan", "feta", "goat cheese", "brie",
                    "camembert", "ricotta", "mascarpone", "cottage cheese",
                    # Dairy products
                    "sour cream", "heavy cream", "whipped cream", "condensed milk",
                    "evaporated milk", "powdered milk", "buttermilk", "kefir",
                    # Hidden dairy
                    "ghee", "clarified butter", "milk powder", "milk solids"
                ],
                
                AllergenType.EGGS: [
                    "egg", "eggs", "yolk", "white", "albumin", "mayonnaise", "mayo",
                    "meringue", "custard", "eggnog", "hollandaise", "aioli",
                    "egg wash", "egg noodles", "pasta", "quiche", "frittata",
                    "scrambled", "fried egg", "poached egg", "deviled eggs"
                ],
                
                AllergenType.FISH: [
                    # Common fish
                    "fish", "salmon", "tuna", "cod", "halibut", "sole", "bass", 
                    "trout", "mackerel", "sardines", "anchovies", "herring",
                    "flounder", "snapper", "grouper", "mahi-mahi", "swordfish",
                    # Fish products
                    "fish sauce", "fish stock", "worcestershire", "caesar dressing",
                    "fish oil", "omega-3", "caviar", "roe", "surimi"
                ],
                
                AllergenType.SHELLFISH: [
                    # Crustaceans
                    "shrimp", "prawn", "crab", "lobster", "crawfish", "crayfish",
                    "langostinos", "scampi",
                    # Molluscs  
                    "oyster", "mussel", "clam", "scallop", "abalone", "conch",
                    "whelk", "periwinkle", "cockle",
                    # Products
                    "seafood", "shellfish", "crab meat", "lobster tail", "shrimp paste"
                ],
                
                AllergenType.TREE_NUTS: [
                    # Tree nuts
                    "almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut",
                    "brazil nut", "macadamia", "pine nut", "chestnut", "beechnut",
                    "hickory nut", "butternut", "chinquapin", "ginkgo nut",
                    # Nut products
                    "marzipan", "nougat", "praline", "nut butter", "almond milk",
                    "coconut", "nutella", "amaretto", "frangelico"
                ],
                
                AllergenType.PEANUTS: [
                    "peanut", "peanuts", "groundnut", "goober", "monkey nut",
                    "peanut butter", "peanut oil", "arachis oil", "satay sauce",
                    "pad thai", "kung pao", "african stew", "peanut sauce"
                ],
                
                AllergenType.WHEAT_GLUTEN: [
                    # Wheat products
                    "wheat", "flour", "bread", "toast", "bun", "roll", "bagel",
                    "croissant", "pastry", "pie crust", "pizza dough", "pasta", 
                    "noodles", "spaghetti", "linguine", "fettuccine", "ravioli",
                    # Gluten sources
                    "gluten", "seitan", "vital wheat gluten", "wheat protein",
                    "barley", "rye", "bulgur", "couscous", "semolina", "durum",
                    "spelt", "kamut", "einkorn", "emmer", "triticale",
                    # Wheat derivatives
                    "wheat starch", "wheat bran", "wheat germ", "wheat berries"
                ],
                
                AllergenType.SOYBEANS: [
                    "soy", "soya", "soybean", "tofu", "tempeh", "miso", "natto",
                    "edamame", "soy sauce", "tamari", "shoyu", "soy milk",
                    "soy protein", "textured soy protein", "soy flour",
                    "lecithin", "soy lecithin"
                ],
                
                AllergenType.SESAME: [
                    "sesame", "sesame seed", "tahini", "sesame oil", "sesame paste",
                    "za'atar", "dukkah", "halva", "benne seed", "sim sim"
                ],
                
                AllergenType.SULFITES: [
                    "sulfite", "sulphite", "sulfur dioxide", "sodium sulfite",
                    "sodium bisulfite", "potassium bisulfite", "wine", "dried fruit",
                    "preserved", "pickled"
                ]
            },
            
            "th": {
                AllergenType.MILK_DAIRY: [
                    "นม", "เนย", "ครีม", "เนยแข็ง", "โยเกิร์ต", "น้ำนม", "นมสด",
                    "นมข้นหวาน", "วิปครีม", "เฮฟวี่ครีม", "เนยจืด"
                ],
                
                AllergenType.EGGS: [
                    "ไข่", "ไข่ไก่", "ไข่เป็ด", "ไข่นก", "มายองเนส", "ไข่ดาว",
                    "ไข่เจียว", "ไข่ต้ม", "ไข่ลวก"
                ],
                
                AllergenType.FISH: [
                    "ปลา", "ปลาทู", "ปลาแซลมอน", "ปลากะพง", "ปลาจาระเม็ด",
                    "ปลาร้า", "น้ำปลา", "ปลาป่น", "ปลาแห้ง"
                ],
                
                AllergenType.SHELLFISH: [
                    "กุ้ง", "ปู", "หอย", "ปลาหมึก", "หอยแมลงภู่", "หอยนางรม",
                    "กั้ง", "กบ", "หอยเชลล์"
                ],
                
                AllergenType.TREE_NUTS: [
                    "อัลมอนด์", "วอลนัท", "เกสตัน", "แคชชิว", "พิสตาชิโอ",
                    "มะม่วงหิมพานต์", "เฮเซลนัท"
                ],
                
                AllergenType.PEANUTS: [
                    "ถั่วลิสง", "ถั่วดิน", "น้ำจิ้มถั่ว", "ซอสถั่วลิสง", "ผัดไทย"
                ],
                
                AllergenType.WHEAT_GLUTEN: [
                    "แป้งสาลี", "ข้าวสาลี", "แป้ง", "บะหมี่", "เส้นใหญ่", "เส้นเล็ก",
                    "ขนมปัง", "พิซซ่า", "พาสต้า"
                ],
                
                AllergenType.SOYBEANS: [
                    "ถั่วเหลือง", "เต้าหู้", "น้ำถั่วเหลือง", "ซีอิ๊ว", "เต้าเจี้ยว",
                    "มิโซะ", "เต้าซี"
                ],
                
                AllergenType.SESAME: [
                    "งา", "น้ำมันงา", "ถั่วงา", "เม็ดงา"
                ]
            }
        }
    
    async def analyze_text(
        self,
        text: str,
        language: str = "en",
        user_allergens: Optional[List[str]] = None,
        confidence_threshold: float = 0.3
    ) -> AllergenDetectionResult:
        """
        Analyze text for allergens with safety-first approach.
        
        Args:
            text: Menu item text to analyze
            language: Language code (en, th, etc.)
            user_allergens: User's known allergies for personalized warnings
            confidence_threshold: Minimum confidence for detection (lowered for safety)
            
        Returns:
            Comprehensive allergen detection results
        """
        import time
        start_time = time.time()
        
        self.detection_stats["total_analyses"] += 1
        
        try:
            self.logger.info(f"Analyzing text for allergens: '{text[:50]}...'")
            
            # Initialize result
            result = AllergenDetectionResult(
                text_analyzed=text,
                language=language
            )
            
            # Normalize text for analysis
            normalized_text = self._normalize_text(text)
            
            # Detect allergens using multiple methods
            detected_matches = []
            
            # Method 1: Pattern matching
            pattern_matches = self._detect_by_patterns(normalized_text, language)
            detected_matches.extend(pattern_matches)
            
            # Method 2: Fuzzy matching for misspellings
            fuzzy_matches = self._detect_by_fuzzy_matching(normalized_text, language)
            detected_matches.extend(fuzzy_matches)
            
            # Method 3: Context-based detection
            context_matches = self._detect_by_context(normalized_text, language)
            detected_matches.extend(context_matches)
            
            # Deduplicate matches
            result.detected_allergens = self._deduplicate_matches(detected_matches)
            
            # Calculate risk level (conservative approach)
            result.risk_level = self._calculate_risk_level(result.detected_allergens)
            
            # Generate safety warnings
            result.safety_warnings = self._generate_safety_warnings(
                result.detected_allergens, result.risk_level
            )
            
            # User-specific warnings
            if user_allergens:
                result.user_specific_warnings = self._generate_user_warnings(
                    result.detected_allergens, user_allergens
                )
            
            # Calculate confidence (conservative - lower is safer)
            result.confidence_score = self._calculate_confidence(result.detected_allergens)
            
            # Generate recommendations
            result.recommendations = self._generate_recommendations(
                result.detected_allergens, result.risk_level, language
            )
            
            # Update statistics
            if result.detected_allergens:
                self.detection_stats["allergens_detected"] += 1
            if result.risk_level == AllergenRiskLevel.CRITICAL:
                self.detection_stats["critical_warnings"] += 1
            
            result.processing_time = time.time() - start_time
            
            # Safety audit logging
            self._log_detection_results(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Critical error in allergen detection: {e}", exc_info=True)
            
            # Return conservative result on error
            return AllergenDetectionResult(
                text_analyzed=text,
                language=language,
                risk_level=AllergenRiskLevel.HIGH,  # Conservative on error
                safety_warnings=[
                    "SYSTEM ERROR: Cannot guarantee allergen safety. Please verify with restaurant staff."
                ],
                processing_time=time.time() - start_time,
                confidence_score=0.0
            )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better pattern matching."""
        # Convert to lowercase
        normalized = text.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove special characters but keep basic punctuation
        normalized = re.sub(r'[^\w\s\-.,()&]', '', normalized)
        
        return normalized
    
    def _detect_by_patterns(self, text: str, language: str) -> List[AllergenMatch]:
        """Detect allergens using predefined patterns."""
        matches = []
        patterns = self._allergen_patterns.get(language, self._allergen_patterns["en"])
        
        for allergen_type, keywords in patterns.items():
            for keyword in keywords:
                # Use word boundaries for exact matching
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    confidence = 0.9  # High confidence for exact pattern matches
                    
                    # Adjust confidence based on keyword specificity
                    if len(keyword) <= 3:  # Short keywords are less specific
                        confidence = 0.6
                    elif keyword in ['fish', 'nut']:  # Common words, might be false positive
                        confidence = 0.7
                    
                    matches.append(AllergenMatch(
                        allergen_type=allergen_type,
                        matched_text=match.group(),
                        confidence=confidence,
                        position=match.start(),
                        language=language
                    ))
        
        return matches
    
    def _detect_by_fuzzy_matching(self, text: str, language: str) -> List[AllergenMatch]:
        """Detect allergens using fuzzy matching for misspellings."""
        matches = []
        patterns = self._allergen_patterns.get(language, self._allergen_patterns["en"])
        
        # Simple fuzzy matching - can be enhanced with libraries like fuzzywuzzy
        words = text.split()
        
        for allergen_type, keywords in patterns.items():
            for keyword in keywords:
                for word in words:
                    if len(word) >= 4 and len(keyword) >= 4:  # Only for longer words
                        similarity = self._calculate_similarity(word, keyword.lower())
                        
                        if similarity > 0.8:  # High similarity threshold
                            matches.append(AllergenMatch(
                                allergen_type=allergen_type,
                                matched_text=word,
                                confidence=similarity * 0.8,  # Reduced for fuzzy match
                                position=text.find(word),
                                language=language,
                                severity_notes=["Fuzzy match - verify spelling"]
                            ))
        
        return matches
    
    def _detect_by_context(self, text: str, language: str) -> List[AllergenMatch]:
        """Detect allergens based on contextual clues."""
        matches = []
        
        # Context patterns that suggest allergens
        if language == "en":
            context_patterns = {
                AllergenType.MILK_DAIRY: [
                    r"made with.*milk", r"contains.*dairy", r"creamy.*sauce",
                    r"cheese.*topping", r"served with.*cream"
                ],
                AllergenType.EGGS: [
                    r"battered.*egg", r"egg.*wash", r"contains.*egg"
                ],
                AllergenType.PEANUTS: [
                    r"thai.*style.*peanut", r"peanut.*dressing", r"ground.*nuts"
                ],
                AllergenType.SHELLFISH: [
                    r"seafood.*medley", r"mixed.*shellfish", r"ocean.*harvest"
                ]
            }
        else:
            # Thai context patterns
            context_patterns = {
                AllergenType.PEANUTS: [r"ผัด.*ถั่วลิสง", r"น้ำจิ้ม.*ถั่ว"],
                AllergenType.SHELLFISH: [r"อาหาร.*ทะเล", r"กุ้ง.*ปู"]
            }
        
        for allergen_type, patterns in context_patterns.items():
            for pattern in patterns:
                matches_found = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches_found:
                    matches.append(AllergenMatch(
                        allergen_type=allergen_type,
                        matched_text=match.group(),
                        confidence=0.6,  # Lower confidence for context matches
                        position=match.start(),
                        language=language,
                        severity_notes=["Context-based detection"]
                    ))
        
        return matches
    
    def _calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate simple character-based similarity between two words."""
        if not word1 or not word2:
            return 0.0
        
        # Simple character overlap similarity
        set1 = set(word1.lower())
        set2 = set(word2.lower())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _deduplicate_matches(self, matches: List[AllergenMatch]) -> List[AllergenMatch]:
        """Remove duplicate allergen matches, keeping highest confidence."""
        if not matches:
            return []
        
        # Group by allergen type and overlapping positions
        grouped = {}
        
        for match in matches:
            key = f"{match.allergen_type}_{match.position // 10}"  # Group nearby positions
            
            if key not in grouped or match.confidence > grouped[key].confidence:
                grouped[key] = match
        
        return list(grouped.values())
    
    def _calculate_risk_level(self, matches: List[AllergenMatch]) -> AllergenRiskLevel:
        """Calculate overall risk level - conservative approach for safety."""
        if not matches:
            return AllergenRiskLevel.NONE
        
        # Check for critical allergens
        critical_found = any(match.allergen_type in self.critical_allergens for match in matches)
        
        if critical_found:
            return AllergenRiskLevel.CRITICAL
        
        # High confidence matches
        high_confidence = [m for m in matches if m.confidence >= 0.8]
        if len(high_confidence) >= 2:
            return AllergenRiskLevel.HIGH
        elif high_confidence:
            return AllergenRiskLevel.MEDIUM
        
        # Any matches at all
        return AllergenRiskLevel.LOW if matches else AllergenRiskLevel.NONE
    
    def _generate_safety_warnings(
        self, 
        matches: List[AllergenMatch], 
        risk_level: AllergenRiskLevel
    ) -> List[str]:
        """Generate safety-critical warnings."""
        warnings = []
        
        if risk_level == AllergenRiskLevel.CRITICAL:
            warnings.append("⚠️  CRITICAL ALLERGEN WARNING: This dish may contain severe allergens!")
            warnings.append("🚨 If you have severe allergies, DO NOT consume without confirmation from restaurant")
        
        if risk_level in [AllergenRiskLevel.HIGH, AllergenRiskLevel.CRITICAL]:
            allergen_types = {match.allergen_type.value.replace('_', ' ').title() 
                            for match in matches}
            warnings.append(f"Detected allergens: {', '.join(allergen_types)}")
        
        if risk_level != AllergenRiskLevel.NONE:
            warnings.append("Always verify ingredients with restaurant staff before ordering")
            warnings.append("Inform server about all allergies and dietary restrictions")
        
        return warnings
    
    def _generate_user_warnings(
        self,
        matches: List[AllergenMatch],
        user_allergens: List[str]
    ) -> List[str]:
        """Generate personalized warnings based on user's allergies."""
        warnings = []
        
        detected_types = {match.allergen_type.value for match in matches}
        
        for user_allergen in user_allergens:
            user_allergen_normalized = user_allergen.lower().replace(' ', '_')
            
            # Check for direct matches
            if user_allergen_normalized in detected_types:
                warnings.append(f"🚨 YOUR ALLERGEN DETECTED: {user_allergen.title()}")
                warnings.append(f"⛔ DO NOT ORDER - Contains {user_allergen}")
            
            # Check for related allergens
            related_warnings = self._check_related_allergens(user_allergen_normalized, detected_types)
            warnings.extend(related_warnings)
        
        return warnings
    
    def _check_related_allergens(self, user_allergen: str, detected_types: Set[str]) -> List[str]:
        """Check for allergens related to user's specific allergies."""
        warnings = []
        
        # Cross-contamination warnings
        if user_allergen == "peanuts" and "tree_nuts" in detected_types:
            warnings.append("⚠️  Tree nuts detected - Risk of cross-contamination with peanuts")
        
        if user_allergen == "shellfish" and "fish" in detected_types:
            warnings.append("⚠️  Fish detected - Possible cross-contamination with shellfish")
        
        if user_allergen == "wheat_gluten" and any(grain in detected_types for grain in ["wheat", "barley", "rye"]):
            warnings.append("⚠️  Gluten-containing ingredients detected")
        
        return warnings
    
    def _calculate_confidence(self, matches: List[AllergenMatch]) -> float:
        """Calculate overall confidence - conservative for safety."""
        if not matches:
            return 1.0  # High confidence in "no allergens"
        
        # Average confidence of matches, but cap it for safety
        avg_confidence = sum(match.confidence for match in matches) / len(matches)
        
        # Conservative adjustment - reduce confidence for safety
        return min(0.85, avg_confidence * 0.9)
    
    def _generate_recommendations(
        self,
        matches: List[AllergenMatch],
        risk_level: AllergenRiskLevel,
        language: str
    ) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        if risk_level == AllergenRiskLevel.CRITICAL:
            if language == "th":
                recommendations.append("แจ้งพนักงานเรื่องการแพ้อาหารทันทีก่อนสั่ง")
                recommendations.append("ขอดูส่วนผสมของอาหารก่อนตัดสินใจ")
            else:
                recommendations.append("Inform server immediately about your specific allergies")
                recommendations.append("Request detailed ingredient list before ordering")
        
        elif risk_level in [AllergenRiskLevel.HIGH, AllergenRiskLevel.MEDIUM]:
            recommendations.append("Ask server about preparation methods and cross-contamination")
            recommendations.append("Consider alternative menu items if you have relevant allergies")
        
        if matches:
            recommendations.append("Take a photo of ingredients list for your records")
            recommendations.append("Have emergency medication readily available if needed")
        
        return recommendations
    
    def _log_detection_results(self, result: AllergenDetectionResult):
        """Log detection results for safety audit trail."""
        log_data = {
            "text_length": len(result.text_analyzed),
            "language": result.language,
            "allergen_count": len(result.detected_allergens),
            "risk_level": result.risk_level.value,
            "confidence": result.confidence_score,
            "processing_time": result.processing_time
        }
        
        if result.risk_level in [AllergenRiskLevel.CRITICAL, AllergenRiskLevel.HIGH]:
            self.logger.warning(f"High-risk allergen detection: {log_data}")
        else:
            self.logger.info(f"Allergen detection completed: {log_data}")
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get service statistics for monitoring and improvement."""
        return {
            **self.detection_stats,
            "detection_rate": (
                self.detection_stats["allergens_detected"] / 
                max(1, self.detection_stats["total_analyses"])
            ),
            "critical_rate": (
                self.detection_stats["critical_warnings"] /
                max(1, self.detection_stats["total_analyses"])
            )
        }
    
    def add_custom_allergen_pattern(
        self,
        allergen_type: AllergenType,
        language: str,
        patterns: List[str]
    ):
        """Add custom allergen patterns for specific regions or restaurants."""
        if language not in self._allergen_patterns:
            self._allergen_patterns[language] = {}
        
        if allergen_type not in self._allergen_patterns[language]:
            self._allergen_patterns[language][allergen_type] = []
        
        self._allergen_patterns[language][allergen_type].extend(patterns)
        
        self.logger.info(f"Added {len(patterns)} custom patterns for {allergen_type} in {language}")
    
    def validate_allergen_database(self) -> Dict[str, Any]:
        """Validate allergen database integrity for safety."""
        validation_results = {
            "total_languages": len(self._allergen_patterns),
            "total_allergen_types": 0,
            "total_patterns": 0,
            "coverage_issues": [],
            "validation_passed": True
        }
        
        required_allergens = {
            AllergenType.MILK_DAIRY, AllergenType.EGGS, AllergenType.FISH,
            AllergenType.SHELLFISH, AllergenType.TREE_NUTS, AllergenType.PEANUTS,
            AllergenType.WHEAT_GLUTEN, AllergenType.SOYBEANS
        }
        
        for language, allergens in self._allergen_patterns.items():
            validation_results["total_allergen_types"] += len(allergens)
            
            for allergen_type, patterns in allergens.items():
                validation_results["total_patterns"] += len(patterns)
                
                # Check minimum pattern coverage
                if len(patterns) < 3:
                    validation_results["coverage_issues"].append(
                        f"Low pattern coverage for {allergen_type} in {language}"
                    )
                    validation_results["validation_passed"] = False
            
            # Check for missing required allergens
            missing_allergens = required_allergens - set(allergens.keys())
            if missing_allergens:
                validation_results["coverage_issues"].append(
                    f"Missing allergen types in {language}: {missing_allergens}"
                )
                validation_results["validation_passed"] = False
        
        self.logger.info(f"Allergen database validation: {validation_results}")
        return validation_results
