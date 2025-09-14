"""Voice navigation templates and instruction formatters.

Provides multilingual turn-by-turn navigation instructions with contextual formatting
for various navigation scenarios and transportation modes.
"""

from enum import Enum
from typing import Dict, List, Optional

from app.schemas.common import LanguageCode
from app.schemas.navigation import TransportMode


class NavigationInstructionType(str, Enum):
    """Types of navigation instructions."""
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    TURN_SLIGHT_LEFT = "turn_slight_left"
    TURN_SLIGHT_RIGHT = "turn_slight_right"
    TURN_SHARP_LEFT = "turn_sharp_left"
    TURN_SHARP_RIGHT = "turn_sharp_right"
    CONTINUE_STRAIGHT = "continue_straight"
    MERGE = "merge"
    ENTER_ROUNDABOUT = "enter_roundabout"
    EXIT_ROUNDABOUT = "exit_roundabout"
    KEEP_LEFT = "keep_left"
    KEEP_RIGHT = "keep_right"
    U_TURN = "u_turn"
    DESTINATION_REACHED = "destination_reached"
    START_ROUTE = "start_route"
    REROUTING = "rerouting"
    SPEED_LIMIT = "speed_limit"
    TRAFFIC_ALERT = "traffic_alert"
    ARRIVE_WAYPOINT = "arrive_waypoint"


class VoiceInstructionTemplates:
    """Multilingual voice instruction templates for navigation."""

    # Base instruction templates by language
    TEMPLATES: Dict[LanguageCode, Dict[NavigationInstructionType, str]] = {
        LanguageCode.EN: {
            NavigationInstructionType.TURN_LEFT: "In {distance}, turn left onto {street}",
            NavigationInstructionType.TURN_RIGHT: "In {distance}, turn right onto {street}",
            NavigationInstructionType.TURN_SLIGHT_LEFT: "In {distance}, keep left onto {street}",
            NavigationInstructionType.TURN_SLIGHT_RIGHT: "In {distance}, keep right onto {street}",
            NavigationInstructionType.TURN_SHARP_LEFT: "In {distance}, make a sharp left onto {street}",
            NavigationInstructionType.TURN_SHARP_RIGHT: "In {distance}, make a sharp right onto {street}",
            NavigationInstructionType.CONTINUE_STRAIGHT: "Continue straight for {distance}",
            NavigationInstructionType.MERGE: "In {distance}, merge onto {street}",
            NavigationInstructionType.ENTER_ROUNDABOUT: "In {distance}, enter the roundabout and take the {exit_number} exit",
            NavigationInstructionType.EXIT_ROUNDABOUT: "Take the {exit_number} exit from the roundabout onto {street}",
            NavigationInstructionType.KEEP_LEFT: "In {distance}, keep left",
            NavigationInstructionType.KEEP_RIGHT: "In {distance}, keep right",
            NavigationInstructionType.U_TURN: "In {distance}, make a U-turn",
            NavigationInstructionType.DESTINATION_REACHED: "You have arrived at your destination",
            NavigationInstructionType.START_ROUTE: "Starting route to {destination}. Total distance: {total_distance}, estimated time: {total_time}",
            NavigationInstructionType.REROUTING: "Rerouting due to traffic conditions",
            NavigationInstructionType.SPEED_LIMIT: "Speed limit: {speed_limit}",
            NavigationInstructionType.TRAFFIC_ALERT: "Traffic alert: {alert_message}",
            NavigationInstructionType.ARRIVE_WAYPOINT: "Arriving at waypoint: {waypoint_name}",
        },
        
        LanguageCode.TH: {
            NavigationInstructionType.TURN_LEFT: "อีก {distance} เลี้ยวซ้ายเข้าสู่ {street}",
            NavigationInstructionType.TURN_RIGHT: "อีก {distance} เลี้ยวขวาเข้าสู่ {street}",
            NavigationInstructionType.TURN_SLIGHT_LEFT: "อีก {distance} แอบซ้ายเข้าสู่ {street}",
            NavigationInstructionType.TURN_SLIGHT_RIGHT: "อีก {distance} แอบขวาเข้าสู่ {street}",
            NavigationInstructionType.TURN_SHARP_LEFT: "อีก {distance} เลี้ยวซ้ายแบบแหลมเข้าสู่ {street}",
            NavigationInstructionType.TURN_SHARP_RIGHT: "อีก {distance} เลี้ยวขวาแบบแหลมเข้าสู่ {street}",
            NavigationInstructionType.CONTINUE_STRAIGHT: "ขับตรงต่อไปอีก {distance}",
            NavigationInstructionType.MERGE: "อีก {distance} ผสานเข้าสู่ {street}",
            NavigationInstructionType.ENTER_ROUNDABOUT: "อีก {distance} เข้าวงเวียนและใช้ทางออกที่ {exit_number}",
            NavigationInstructionType.EXIT_ROUNDABOUT: "ใช้ทางออกที่ {exit_number} จากวงเวียนเข้าสู่ {street}",
            NavigationInstructionType.KEEP_LEFT: "อีก {distance} ชิดซ้าย",
            NavigationInstructionType.KEEP_RIGHT: "อีก {distance} ชิดขวา",
            NavigationInstructionType.U_TURN: "อีก {distance} ยูเทิร์น",
            NavigationInstructionType.DESTINATION_REACHED: "คุณมาถึงปลายทางแล้ว",
            NavigationInstructionType.START_ROUTE: "เริ่มเส้นทางไป {destination} ระยะทางรวม: {total_distance} เวลาโดยประมาณ: {total_time}",
            NavigationInstructionType.REROUTING: "กำลังหาเส้นทางใหม่เนื่องจากสภาพการจราจร",
            NavigationInstructionType.SPEED_LIMIT: "ขีดจำกัดความเร็ว: {speed_limit}",
            NavigationInstructionType.TRAFFIC_ALERT: "แจ้งเตือนการจราจร: {alert_message}",
            NavigationInstructionType.ARRIVE_WAYPOINT: "มาถึงจุดพัก: {waypoint_name}",
        },
        
        LanguageCode.ES: {
            NavigationInstructionType.TURN_LEFT: "En {distance}, gire a la izquierda hacia {street}",
            NavigationInstructionType.TURN_RIGHT: "En {distance}, gire a la derecha hacia {street}",
            NavigationInstructionType.TURN_SLIGHT_LEFT: "En {distance}, manténgase a la izquierda hacia {street}",
            NavigationInstructionType.TURN_SLIGHT_RIGHT: "En {distance}, manténgase a la derecha hacia {street}",
            NavigationInstructionType.TURN_SHARP_LEFT: "En {distance}, gire bruscamente a la izquierda hacia {street}",
            NavigationInstructionType.TURN_SHARP_RIGHT: "En {distance}, gire bruscamente a la derecha hacia {street}",
            NavigationInstructionType.CONTINUE_STRAIGHT: "Continúe recto por {distance}",
            NavigationInstructionType.MERGE: "En {distance}, incorpórese a {street}",
            NavigationInstructionType.ENTER_ROUNDABOUT: "En {distance}, entre en la rotonda y tome la {exit_number} salida",
            NavigationInstructionType.EXIT_ROUNDABOUT: "Tome la {exit_number} salida de la rotonda hacia {street}",
            NavigationInstructionType.KEEP_LEFT: "En {distance}, manténgase a la izquierda",
            NavigationInstructionType.KEEP_RIGHT: "En {distance}, manténgase a la derecha",
            NavigationInstructionType.U_TURN: "En {distance}, dé la vuelta",
            NavigationInstructionType.DESTINATION_REACHED: "Ha llegado a su destino",
            NavigationInstructionType.START_ROUTE: "Iniciando ruta hacia {destination}. Distancia total: {total_distance}, tiempo estimado: {total_time}",
            NavigationInstructionType.REROUTING: "Recalculando ruta debido a las condiciones del tráfico",
            NavigationInstructionType.SPEED_LIMIT: "Límite de velocidad: {speed_limit}",
            NavigationInstructionType.TRAFFIC_ALERT: "Alerta de tráfico: {alert_message}",
            NavigationInstructionType.ARRIVE_WAYPOINT: "Llegando al punto de referencia: {waypoint_name}",
        },
        
        LanguageCode.FR: {
            NavigationInstructionType.TURN_LEFT: "Dans {distance}, tournez à gauche sur {street}",
            NavigationInstructionType.TURN_RIGHT: "Dans {distance}, tournez à droite sur {street}",
            NavigationInstructionType.TURN_SLIGHT_LEFT: "Dans {distance}, serrez à gauche sur {street}",
            NavigationInstructionType.TURN_SLIGHT_RIGHT: "Dans {distance}, serrez à droite sur {street}",
            NavigationInstructionType.TURN_SHARP_LEFT: "Dans {distance}, tournez brusquement à gauche sur {street}",
            NavigationInstructionType.TURN_SHARP_RIGHT: "Dans {distance}, tournez brusquement à droite sur {street}",
            NavigationInstructionType.CONTINUE_STRAIGHT: "Continuez tout droit pendant {distance}",
            NavigationInstructionType.MERGE: "Dans {distance}, fusionnez sur {street}",
            NavigationInstructionType.ENTER_ROUNDABOUT: "Dans {distance}, entrez dans le rond-point et prenez la {exit_number} sortie",
            NavigationInstructionType.EXIT_ROUNDABOUT: "Prenez la {exit_number} sortie du rond-point vers {street}",
            NavigationInstructionType.KEEP_LEFT: "Dans {distance}, restez à gauche",
            NavigationInstructionType.KEEP_RIGHT: "Dans {distance}, restez à droite",
            NavigationInstructionType.U_TURN: "Dans {distance}, faites demi-tour",
            NavigationInstructionType.DESTINATION_REACHED: "Vous êtes arrivé à destination",
            NavigationInstructionType.START_ROUTE: "Démarrage de l'itinéraire vers {destination}. Distance totale: {total_distance}, durée estimée: {total_time}",
            NavigationInstructionType.REROUTING: "Recalcul de l'itinéraire en raison des conditions de circulation",
            NavigationInstructionType.SPEED_LIMIT: "Limitation de vitesse: {speed_limit}",
            NavigationInstructionType.TRAFFIC_ALERT: "Alerte circulation: {alert_message}",
            NavigationInstructionType.ARRIVE_WAYPOINT: "Arrivée au point d'étape: {waypoint_name}",
        }
    }

    # Distance formatting by language
    DISTANCE_FORMATS: Dict[LanguageCode, Dict[str, str]] = {
        LanguageCode.EN: {
            "meters": "{value} meters",
            "kilometers": "{value} kilometers", 
            "feet": "{value} feet",
            "miles": "{value} miles",
            "now": "now",
            "soon": "soon"
        },
        LanguageCode.TH: {
            "meters": "{value} เมตร",
            "kilometers": "{value} กิโลเมตร",
            "feet": "{value} ฟุต",
            "miles": "{value} ไมล์",
            "now": "ตอนนี้",
            "soon": "เร็วๆ นี้"
        },
        LanguageCode.ES: {
            "meters": "{value} metros",
            "kilometers": "{value} kilómetros",
            "feet": "{value} pies", 
            "miles": "{value} millas",
            "now": "ahora",
            "soon": "pronto"
        },
        LanguageCode.FR: {
            "meters": "{value} mètres",
            "kilometers": "{value} kilomètres",
            "feet": "{value} pieds",
            "miles": "{value} miles",
            "now": "maintenant",
            "soon": "bientôt"
        }
    }

    @classmethod
    def get_instruction_template(
        cls,
        instruction_type: NavigationInstructionType,
        language: LanguageCode = LanguageCode.EN
    ) -> str:
        """Get instruction template for given type and language.
        
        Args:
            instruction_type: The type of navigation instruction
            language: Target language for the instruction
            
        Returns:
            Formatted instruction template string
        """
        return cls.TEMPLATES.get(language, cls.TEMPLATES[LanguageCode.EN]).get(
            instruction_type, 
            cls.TEMPLATES[LanguageCode.EN][instruction_type]
        )

    @classmethod
    def format_distance(
        cls,
        distance_meters: float,
        language: LanguageCode = LanguageCode.EN,
        use_imperial: bool = False
    ) -> str:
        """Format distance for voice instructions.
        
        Args:
            distance_meters: Distance in meters
            language: Target language 
            use_imperial: Whether to use imperial units (feet/miles)
            
        Returns:
            Formatted distance string in appropriate language and units
        """
        distance_formats = cls.DISTANCE_FORMATS.get(language, cls.DISTANCE_FORMATS[LanguageCode.EN])
        
        if distance_meters < 10:
            return distance_formats["now"]
        elif distance_meters < 50:
            return distance_formats["soon"]
        elif use_imperial:
            if distance_meters < 304.8:  # Less than 1000 feet
                feet = round(distance_meters * 3.28084)
                return distance_formats["feet"].format(value=feet)
            else:
                miles = round(distance_meters * 0.000621371, 1)
                return distance_formats["miles"].format(value=miles)
        else:
            if distance_meters < 1000:
                meters = round(distance_meters)
                return distance_formats["meters"].format(value=meters)
            else:
                kilometers = round(distance_meters / 1000, 1)
                return distance_formats["kilometers"].format(value=kilometers)

    @classmethod
    def create_voice_instruction(
        cls,
        instruction_type: NavigationInstructionType,
        language: LanguageCode = LanguageCode.EN,
        distance_meters: Optional[float] = None,
        street_name: Optional[str] = None,
        exit_number: Optional[int] = None,
        destination: Optional[str] = None,
        total_distance: Optional[str] = None,
        total_time: Optional[str] = None,
        speed_limit: Optional[str] = None,
        alert_message: Optional[str] = None,
        waypoint_name: Optional[str] = None,
        use_imperial: bool = False
    ) -> str:
        """Create a formatted voice instruction.
        
        Args:
            instruction_type: Type of navigation instruction
            language: Target language for instruction
            distance_meters: Distance to maneuver in meters
            street_name: Name of target street/road
            exit_number: Roundabout exit number 
            destination: Destination name for route start
            total_distance: Total route distance for route start
            total_time: Total route time for route start
            speed_limit: Speed limit value
            alert_message: Traffic alert message
            waypoint_name: Waypoint name
            use_imperial: Whether to use imperial units
            
        Returns:
            Formatted voice instruction ready for TTS
        """
        template = cls.get_instruction_template(instruction_type, language)
        
        # Prepare replacement values
        replacements = {}
        
        if distance_meters is not None:
            replacements["distance"] = cls.format_distance(distance_meters, language, use_imperial)
            
        if street_name:
            replacements["street"] = street_name
            
        if exit_number is not None:
            # Format exit number based on language
            if language == LanguageCode.TH:
                replacements["exit_number"] = f"{exit_number}"
            elif language == LanguageCode.ES:
                replacements["exit_number"] = f"{exit_number}ª" if exit_number == 1 else f"{exit_number}ª"
            elif language == LanguageCode.FR:
                replacements["exit_number"] = f"{exit_number}ème" if exit_number > 1 else f"{exit_number}ère"
            else:  # English and fallback
                suffix = "st" if exit_number == 1 else "nd" if exit_number == 2 else "rd" if exit_number == 3 else "th"
                replacements["exit_number"] = f"{exit_number}{suffix}"
        
        if destination:
            replacements["destination"] = destination
            
        if total_distance:
            replacements["total_distance"] = total_distance
            
        if total_time:
            replacements["total_time"] = total_time
            
        if speed_limit:
            replacements["speed_limit"] = speed_limit
            
        if alert_message:
            replacements["alert_message"] = alert_message
            
        if waypoint_name:
            replacements["waypoint_name"] = waypoint_name
        
        # Format template with available replacements
        try:
            return template.format(**replacements)
        except KeyError as e:
            # Return template with missing placeholders for debugging
            return template

    @classmethod
    def get_supported_languages(cls) -> List[LanguageCode]:
        """Get list of supported languages for voice instructions.
        
        Returns:
            List of supported language codes
        """
        return list(cls.TEMPLATES.keys())

    @classmethod
    def detect_instruction_type(cls, maneuver: str) -> NavigationInstructionType:
        """Detect instruction type from Google Maps maneuver string.
        
        Args:
            maneuver: Google Maps maneuver identifier
            
        Returns:
            Corresponding NavigationInstructionType
        """
        maneuver_mapping = {
            "turn-left": NavigationInstructionType.TURN_LEFT,
            "turn-right": NavigationInstructionType.TURN_RIGHT,
            "turn-slight-left": NavigationInstructionType.TURN_SLIGHT_LEFT,
            "turn-slight-right": NavigationInstructionType.TURN_SLIGHT_RIGHT,
            "turn-sharp-left": NavigationInstructionType.TURN_SHARP_LEFT,
            "turn-sharp-right": NavigationInstructionType.TURN_SHARP_RIGHT,
            "straight": NavigationInstructionType.CONTINUE_STRAIGHT,
            "merge": NavigationInstructionType.MERGE,
            "roundabout-left": NavigationInstructionType.ENTER_ROUNDABOUT,
            "roundabout-right": NavigationInstructionType.ENTER_ROUNDABOUT,
            "keep-left": NavigationInstructionType.KEEP_LEFT,
            "keep-right": NavigationInstructionType.KEEP_RIGHT,
            "uturn-left": NavigationInstructionType.U_TURN,
            "uturn-right": NavigationInstructionType.U_TURN,
        }
        
        return maneuver_mapping.get(maneuver, NavigationInstructionType.CONTINUE_STRAIGHT)


class VoiceNavigationService:
    """Service for managing voice-guided navigation."""
    
    def __init__(self):
        """Initialize voice navigation service."""
        self.templates = VoiceInstructionTemplates()
        self.active_voice_settings = {}  # session_id -> voice settings
    
    def set_voice_settings(
        self,
        session_id: str,
        language: LanguageCode = LanguageCode.EN,
        use_imperial: bool = False,
        voice_speed: float = 1.0,
        announce_distance_threshold: int = 500  # meters
    ) -> None:
        """Set voice settings for a navigation session.
        
        Args:
            session_id: Navigation session identifier
            language: Voice instruction language
            use_imperial: Whether to use imperial units
            voice_speed: TTS speech rate multiplier
            announce_distance_threshold: Distance threshold for announcements
        """
        self.active_voice_settings[session_id] = {
            "language": language,
            "use_imperial": use_imperial,
            "voice_speed": voice_speed,
            "announce_distance_threshold": announce_distance_threshold
        }
    
    def get_voice_instruction_for_step(
        self,
        session_id: str,
        maneuver: str,
        distance_meters: float,
        street_name: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """Generate voice instruction for a navigation step.
        
        Args:
            session_id: Navigation session identifier
            maneuver: Navigation maneuver type
            distance_meters: Distance to maneuver
            street_name: Target street name
            **kwargs: Additional instruction parameters
            
        Returns:
            Formatted voice instruction or None if not ready to announce
        """
        settings = self.active_voice_settings.get(session_id, {
            "language": LanguageCode.EN,
            "use_imperial": False,
            "announce_distance_threshold": 500
        })
        
        # Check if we should announce based on distance
        if distance_meters > settings["announce_distance_threshold"]:
            return None
        
        instruction_type = self.templates.detect_instruction_type(maneuver)
        
        return self.templates.create_voice_instruction(
            instruction_type=instruction_type,
            language=settings["language"],
            distance_meters=distance_meters,
            street_name=street_name,
            use_imperial=settings["use_imperial"],
            **kwargs
        )
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up voice settings for a completed session.
        
        Args:
            session_id: Navigation session identifier
        """
        self.active_voice_settings.pop(session_id, None)
