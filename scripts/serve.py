import uvicorn

from ragnostics.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        app="ragnostics.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
