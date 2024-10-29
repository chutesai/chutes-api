from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from chute.router import router as chute_router
from image.router import router as image_router
from database import Base, engine

app = FastAPI(default_response_class=ORJSONResponse)

app.include_router(chute_router, prefix="/chutes", tags=["Chutes"])
app.include_router(image_router, prefix="/images", tags=["Images"])


@app.on_event("startup")
async def startup():
    """
    Ensure tables are initialized.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
