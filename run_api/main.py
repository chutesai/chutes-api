from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from run_api.chute.router import router as chute_router
from run_api.image.router import router as image_router
from run_api.database import Base, engine
from run_api.config import settings

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

    if not await settings.storage_client.bucket_exists(settings.storage_bucket):
        await settings.storage_client.make_bucket(settings.storage_bucket)
