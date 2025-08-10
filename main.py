import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app:app",  # app.py -> app object
        host="0.0.0.0",
        port=int(__import__("os").environ.get("PORT", 8080)),
        reload=False
    )
