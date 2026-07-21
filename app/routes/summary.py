from fastapi import APIRouter, HTTPException
from app.models import SummaryRequest, SummaryResponse
from app.services.material_service import MaterialService
from app.services.summary_service import SummaryService

router = APIRouter(prefix="/api/v1/summary", tags=["summary"])
material_service = MaterialService()
summary_service = SummaryService()

@router.post("/generate", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    """
    Generate a topic-structured summary of course material
    
    - **slug**: The course material slug (e.g., "formation-of-a-contract-offer-acceptance")
    - **max_words**: Maximum total words for the summary (default: 500, min: 100, max: 2000)
    """
    try:
        # Fetch material from the API
        material_data = await material_service.fetch_material(request.slug)
        
        # Generate summary
        summary_data = await summary_service.generate_summary(
            material_data, 
            request.max_words
        )
        
        return SummaryResponse(
            success=True,
            module_title=summary_data["module_title"],
            module_description=summary_data["module_description"],
            topics=summary_data["topics"],
            total_word_count=summary_data["total_word_count"],
            summary_word_count=summary_data["summary_word_count"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")