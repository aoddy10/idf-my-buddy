"""Feature flags management for My Buddy application.

This module provides a centralized way to manage feature flags for
gradual rollouts, A/B testing, and environment-specific features.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from app.core.config import settings


class FeatureFlag(str, Enum):
    """Available feature flags."""
    
    # AI Services
    OFFLINE_MODE = "offline_mode"
    CLOUD_FALLBACK = "cloud_fallback"
    WHISPER_OPTIMIZATION = "whisper_optimization"
    NLLB_CACHING = "nllb_caching"
    
    # Core Features  
    NAVIGATION_ASSISTANT = "navigation_assistant"
    RESTAURANT_ASSISTANT = "restaurant_assistant"
    SHOPPING_ASSISTANT = "shopping_assistant"
    SAFETY_FEATURES = "safety_features"
    
    # Advanced Features
    AR_OVERLAY = "ar_overlay"
    VOICE_COMMANDS = "voice_commands"
    ALLERGEN_DETECTION = "allergen_detection"
    CULTURAL_CONTEXT = "cultural_context"
    
    # Monitoring & Analytics
    TELEMETRY = "telemetry"
    PERFORMANCE_MONITORING = "performance_monitoring"
    ERROR_REPORTING = "error_reporting"
    
    # Development & Debug
    DEBUG_ENDPOINTS = "debug_endpoints"
    MOCK_SERVICES = "mock_services"
    SYNTHETIC_DATA = "synthetic_data"


@dataclass
class FeatureFlagConfig:
    """Configuration for a feature flag."""
    
    enabled: bool
    rollout_percentage: float = 100.0
    user_groups: Optional[list] = None
    environment_restrictions: Optional[list] = None
    description: str = ""


class FeatureFlagManager:
    """Manages feature flags and rollout logic."""
    
    def __init__(self) -> None:
        """Initialize feature flag manager with default configurations."""
        self._flags: Dict[FeatureFlag, FeatureFlagConfig] = {
            # AI Services - controlled by environment settings
            FeatureFlag.OFFLINE_MODE: FeatureFlagConfig(
                enabled=settings.enable_offline_mode,
                description="Enable offline AI model processing"
            ),
            FeatureFlag.CLOUD_FALLBACK: FeatureFlagConfig(
                enabled=settings.enable_cloud_fallback,
                description="Enable cloud API fallback for AI services"
            ),
            FeatureFlag.WHISPER_OPTIMIZATION: FeatureFlagConfig(
                enabled=True,
                description="Enable Whisper model optimizations"
            ),
            FeatureFlag.NLLB_CACHING: FeatureFlagConfig(
                enabled=True,
                description="Enable NLLB translation caching"
            ),
            
            # Core Features - production ready
            FeatureFlag.NAVIGATION_ASSISTANT: FeatureFlagConfig(
                enabled=True,
                description="Navigation and wayfinding assistance"
            ),
            FeatureFlag.RESTAURANT_ASSISTANT: FeatureFlagConfig(
                enabled=True,
                description="Restaurant and menu assistance"
            ),
            FeatureFlag.SHOPPING_ASSISTANT: FeatureFlagConfig(
                enabled=True,
                description="Shopping and product assistance"
            ),
            FeatureFlag.SAFETY_FEATURES: FeatureFlagConfig(
                enabled=settings.enable_safety_features,
                description="Safety and emergency features"
            ),
            
            # Advanced Features - gradual rollout
            FeatureFlag.AR_OVERLAY: FeatureFlagConfig(
                enabled=False,
                rollout_percentage=25.0,
                description="Augmented reality overlay features"
            ),
            FeatureFlag.VOICE_COMMANDS: FeatureFlagConfig(
                enabled=True,
                rollout_percentage=75.0,
                description="Voice command processing"
            ),
            FeatureFlag.ALLERGEN_DETECTION: FeatureFlagConfig(
                enabled=True,
                rollout_percentage=50.0,
                description="Food allergen detection and warnings"
            ),
            FeatureFlag.CULTURAL_CONTEXT: FeatureFlagConfig(
                enabled=False,
                rollout_percentage=10.0,
                description="Cultural context and etiquette suggestions"
            ),
            
            # Monitoring & Analytics
            FeatureFlag.TELEMETRY: FeatureFlagConfig(
                enabled=settings.enable_telemetry,
                description="Anonymous usage telemetry"
            ),
            FeatureFlag.PERFORMANCE_MONITORING: FeatureFlagConfig(
                enabled=not settings.is_development,
                description="Performance metrics collection"
            ),
            FeatureFlag.ERROR_REPORTING: FeatureFlagConfig(
                enabled=settings.is_production,
                description="Automated error reporting"
            ),
            
            # Development & Debug
            FeatureFlag.DEBUG_ENDPOINTS: FeatureFlagConfig(
                enabled=settings.enable_debug_endpoints and settings.is_development,
                environment_restrictions=["development", "testing"],
                description="Development debug endpoints"
            ),
            FeatureFlag.MOCK_SERVICES: FeatureFlagConfig(
                enabled=settings.is_development,
                environment_restrictions=["development", "testing"],
                description="Use mock services instead of real APIs"
            ),
            FeatureFlag.SYNTHETIC_DATA: FeatureFlagConfig(
                enabled=settings.is_development,
                environment_restrictions=["development", "testing"],
                description="Generate synthetic test data"
            ),
        }
    
    def is_enabled(
        self,
        flag: FeatureFlag,
        user_id: Optional[str] = None,
        user_groups: Optional[list] = None
    ) -> bool:
        """Check if a feature flag is enabled for a user.
        
        Args:
            flag: Feature flag to check.
            user_id: Optional user ID for rollout percentage.
            user_groups: Optional user groups for targeting.
            
        Returns:
            bool: True if feature is enabled for the user.
        """
        config = self._flags.get(flag)
        if not config:
            return False
        
        # Check basic enabled status
        if not config.enabled:
            return False
        
        # Check environment restrictions
        if config.environment_restrictions:
            if settings.app_env not in config.environment_restrictions:
                return False
        
        # Check user group restrictions
        if config.user_groups and user_groups:
            if not any(group in config.user_groups for group in user_groups):
                return False
        
        # Check rollout percentage
        if config.rollout_percentage < 100.0:
            if user_id:
                # Use hash of user_id for consistent rollout
                import hashlib
                hash_value = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
                user_percentage = (hash_value % 100) + 1
                return user_percentage <= config.rollout_percentage
            else:
                # No user ID, use random rollout
                import random
                return random.random() * 100 <= config.rollout_percentage
        
        return True
    
    def get_flag_config(self, flag: FeatureFlag) -> Optional[FeatureFlagConfig]:
        """Get configuration for a specific flag.
        
        Args:
            flag: Feature flag to get configuration for.
            
        Returns:
            FeatureFlagConfig: Flag configuration or None if not found.
        """
        return self._flags.get(flag)
    
    def update_flag(
        self,
        flag: FeatureFlag,
        enabled: Optional[bool] = None,
        rollout_percentage: Optional[float] = None
    ) -> None:
        """Update feature flag configuration at runtime.
        
        Args:
            flag: Feature flag to update.
            enabled: New enabled status.
            rollout_percentage: New rollout percentage.
        """
        config = self._flags.get(flag)
        if not config:
            return
        
        if enabled is not None:
            config.enabled = enabled
        
        if rollout_percentage is not None:
            config.rollout_percentage = max(0.0, min(100.0, rollout_percentage))
    
    def get_enabled_flags(
        self,
        user_id: Optional[str] = None,
        user_groups: Optional[list] = None
    ) -> Dict[str, bool]:
        """Get all enabled flags for a user.
        
        Args:
            user_id: Optional user ID for rollout percentage.
            user_groups: Optional user groups for targeting.
            
        Returns:
            Dict[str, bool]: Map of flag names to enabled status.
        """
        return {
            flag.value: self.is_enabled(flag, user_id, user_groups)
            for flag in FeatureFlag
        }


# Global feature flag manager instance
feature_flags = FeatureFlagManager()


def is_feature_enabled(
    flag: FeatureFlag,
    user_id: Optional[str] = None,
    user_groups: Optional[list] = None
) -> bool:
    """Check if a feature flag is enabled (convenience function).
    
    Args:
        flag: Feature flag to check.
        user_id: Optional user ID for rollout percentage.
        user_groups: Optional user groups for targeting.
        
    Returns:
        bool: True if feature is enabled for the user.
    """
    return feature_flags.is_enabled(flag, user_id, user_groups)
