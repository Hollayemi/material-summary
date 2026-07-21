import httpx
from typing import Dict, Any, Optional
from fastapi import HTTPException

BASE_URL = "https://lawticha.onrender.com/api/v1/learn/material"

class MaterialService:
    @staticmethod
    async def fetch_material(slug: str) -> Dict[str, Any]:
        """Fetch course material from the API"""
        url = f"{BASE_URL}/{slug}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                if not data.get("success"):
                    raise HTTPException(
                        status_code=400,
                        detail=data.get("message", "Failed to retrieve material")
                    )
                
                return data.get("data", {}).get("data", {})
                
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Request to material API timed out")
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Material API error: {e.response.text}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")