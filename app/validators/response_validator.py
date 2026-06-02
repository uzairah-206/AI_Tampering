from app.models.response_models import GeminiResponseModel

class ResponseValidator:
    @classmethod
    def validate(cls, raw_json: dict) -> GeminiResponseModel:
        if not raw_json:
            raise ValueError("Empty response received from API.")
            
        try:
            validated_model = GeminiResponseModel(**raw_json)
            return validated_model
        except Exception as e:
            raise ValueError(f"Malformed response schema: {e}")
