## FEATURE:

AI Buddy is a multimodal AI assistant designed to support users in navigation, restaurant recommendations, shopping assistance, and personal safety. It leverages edge AI capabilities including OCR (Optical Character Recognition), ASR (Automatic Speech Recognition), MT (Machine Translation), and TTS (Text-to-Speech) to provide seamless interaction. The system integrates with map services for real-time location and routing, supports personalization to adapt to user preferences, and includes panic and safety tools for emergency situations.

## EXAMPLES:

See the `examples/` folder for demonstration scripts and modules. Expected examples include CLI demos showcasing core functionalities, feature-specific modules such as navigation and restaurant recommendation workflows, and best practice implementations illustrating integration patterns and usage guidelines.

## DOCUMENTATION:

Key documentation sources to reference during development include:

-   [Pydantic AI](https://ai.pydantic.dev/) for data validation and settings management
-   [FastAPI](https://fastapi.tiangolo.com/) for backend API development
-   [SQLModel](https://sqlmodel.tiangolo.com/) for database modeling and interaction
-   [React Native](https://reactnative.dev/) and [Flutter](https://flutter.dev/docs) for cross-platform mobile app development
-   [Whisper](https://github.com/openai/whisper) for speech recognition
-   [NLLB (No Language Left Behind)](https://ai.meta.com/research/no-language-left-behind/) for machine translation
-   [ML Kit](https://developers.google.com/ml-kit) and [Apple Vision](https://developer.apple.com/documentation/vision/) for on-device computer vision tasks
-   [Mapbox](https://docs.mapbox.com/) for mapping and location services

## OTHER CONSIDERATIONS:

-   Include a `.env.example` file with placeholders for API keys and configuration variables.
-   Provide clear README setup instructions covering map API key configuration and cloud/edge model deployment.
-   Ensure multilingual and internationalization (i18n) support throughout the application.
-   Adopt a privacy-first design approach to protect user data.
-   Maintain performance budgets to ensure responsive and efficient operation.
-   Enforce test coverage with automated testing strategies.
-   Manage environment variables securely using python_dotenv or equivalent tools.
