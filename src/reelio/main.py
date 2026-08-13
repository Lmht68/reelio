from fastapi import FastAPI

app = FastAPI(
    title="Reelio API",
    version="0.1.0",
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Reelio API is running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
