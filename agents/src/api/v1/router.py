import logging

from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse

from crew import BarcodeLookupCrew
from src.crews.product_analysis_crew import ProductAnalysisCrew

from src.api.v1.schemas import BarcodeRequest, BarcodeResponse, RAGQueryRequest, RAGQueryResponse

router = APIRouter()

logger = logging.getLogger(__name__)

@router.post("/search_barcode")
async def search_barcode(request: BarcodeRequest) -> BarcodeResponse:
    barcode = request.barcode.strip()
    if not barcode.isdigit():
        raise HTTPException(status_code=400, detail="Barcode must contain only digits")

    try:

        inputs = {"barcode": barcode}
        result = BarcodeLookupCrew().crew().kickoff(inputs=inputs)

        response_result = BarcodeResponse(
            barcode=barcode,
            product_info=str(result)
        )
        return response_result
    except Exception as e:
        logging.error(f"Error during barcode lookup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process barcode")


@router.post("/analyze_product")
async def analyze_product(request: RAGQueryRequest) -> JSONResponse:
    if not request.product_info or not request.product_info.strip():
        raise HTTPException(status_code=400, detail="Product info cannot be empty")

    try:
        crew_inputs = {
            "product_info": request.product_info,
            "collection_id": request.collection_id,
            "system_prompt": request.system_prompt,
            "analysis_type": request.analysis_type
        }

        crew_result = ProductAnalysisCrew().crew(
            analysis_type=request.analysis_type
        ).kickoff(inputs=crew_inputs)

        if request.analysis_type == "summary":
            return JSONResponse(content={"summary": crew_result.raw})

        elif hasattr(crew_result, 'pydantic') and crew_result.pydantic:
            return JSONResponse(content=crew_result.pydantic.model_dump())

        elif hasattr(crew_result, 'json_dict') and crew_result.json_dict:
            return JSONResponse(content=crew_result.json_dict)

        else:
            return JSONResponse(content={"result": str(crew_result.raw)})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during product analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to perform product analysis: {str(e)}")