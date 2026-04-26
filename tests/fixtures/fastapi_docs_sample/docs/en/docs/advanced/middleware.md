---
title: Advanced Middleware
---

# Advanced Middleware

You can add middleware to **FastAPI** applications.

A "middleware" is a function that works with every **request** before it is processed
by any specific path operation, and with every **response** before returning it.

```python
import time

from fastapi import FastAPI, Request

app = FastAPI()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```
