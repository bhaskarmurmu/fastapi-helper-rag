# First Steps

The simplest FastAPI file looks like this:

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
```

Copy that to a file `main.py`.
