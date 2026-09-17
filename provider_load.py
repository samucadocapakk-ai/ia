"""Share one authenticated Featherless SSE connection across all visitors."""
import asyncio
import contextlib
import json
import math
import os
import time

import httpx
from fastapi.responses import StreamingResponse

URL = "https://api.featherless.ai/account/concurrency/stream"


def aggregate(payload):
    """Show weighted usage against this app's six-unit display allocation."""
    if not isinstance(payload, dict):
        raise ValueError("Invalid concurrency snapshot")
    values = {}
    for field in ("limit", "used_cost", "request_count"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Missing numeric concurrency field")
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid concurrency value")
        values[field] = value
    if values["limit"] <= 0:
        raise ValueError("No concurrency capacity available")
    return {"status": "live", **values, "limit": 6,
            "percent": round(min(100, 100 * values["used_cost"] / 6), 1),
            "updated_at": time.time()}


async def snapshots(lines):
    """Parse standard SSE frames, including comments and multiline data."""
    data = []
    size = 0
    async for line in lines:
        if line == "":
            if data:
                try:
                    yield aggregate(json.loads("\n".join(data)))
                except (ValueError, TypeError):
                    pass
            data, size = [], 0
        elif line.startswith("data:"):
            value = line[5:].removeprefix(" ")
            size += len(value)
            if size > 1_000_000:
                raise ValueError("Oversized provider event")
            data.append(value)


class LoadMonitor:
    def __init__(self):
        self.latest = {"status": "connecting"}
        self.subscribers = set()
        self.task = None

    def publish(self, value):
        self.latest = value
        for queue in tuple(self.subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(value)

    def subscribe(self):
        queue = asyncio.Queue(maxsize=1)
        self.subscribers.add(queue)
        queue.put_nowait(self.latest)
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.run())
        return queue

    async def run(self):
        delay = 2
        while True:
            key = os.environ.get("FEATHERLESS_API_KEY", "").strip()
            if not key:
                self.publish({"status": "unavailable"})
                await asyncio.sleep(30)
                continue
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10)) as client:
                    async with client.stream("GET", URL, headers={
                        "Authorization": "Bearer " + key,
                        "Accept": "text/event-stream",
                        "Cache-Control": "no-cache",
                    }) as response:
                        response.raise_for_status()
                        async with contextlib.aclosing(snapshots(response.aiter_lines())) as feed:
                            while True:
                                try:
                                    value = await asyncio.wait_for(anext(feed), timeout=35)
                                except StopAsyncIteration:
                                    break
                                delay = 2
                                self.publish(value)
                # A clean EOF is also a disconnected feed, never a zero load.
                self.publish({"status": "reconnecting"})
            except asyncio.CancelledError:
                raise
            except Exception:
                # Never forward upstream bodies, headers, account data or keys.
                self.publish({"status": "unavailable"})
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)

    async def close(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
            self.task = None


def install_load_meter(app):
    monitor = LoadMonitor()
    app.state.provider_load = monitor
    previous_lifespan = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(application):
        async with previous_lifespan(application) as state:
            try:
                yield state
            finally:
                await monitor.close()

    app.router.lifespan_context = lifespan

    @app.get("/api/system-load/stream")
    async def stream_load():
        async def events():
            queue = monitor.subscribe()
            try:
                yield "retry: 3000\n\n"
                while True:
                    try:
                        value = await asyncio.wait_for(queue.get(), timeout=10)
                    except asyncio.TimeoutError:
                        # Keep proxies alive, but do not refresh the data timestamp.
                        yield ": keepalive\n\n"
                        continue
                    yield "data: " + json.dumps(value) + "\n\n"
            finally:
                monitor.subscribers.discard(queue)
        return StreamingResponse(events(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no",
        })
