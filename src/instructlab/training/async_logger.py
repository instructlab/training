# Standard
from datetime import datetime
import asyncio
import json
import threading

# Third Party
import aiofiles


class AsyncStructuredLogger:
    def __init__(self, file_name="training_log.jsonl"):
        self.file_name = file_name
        self.logs = []
        self.loop = asyncio.new_event_loop()
        t = threading.Thread(
            target=self._run_event_loop, args=(self.loop,), daemon=True
        )
        t.start()
        asyncio.run_coroutine_threadsafe(self._initialize_log_file(), self.loop)

    def _run_event_loop(self, loop):
        asyncio.set_event_loop(loop)  #
        loop.run_forever()

    async def _initialize_log_file(self):
        self.logs = []
        try:
            async with aiofiles.open(self.file_name, "r") as f:
                async for line in f:
                    if line.strip():  # Avoid empty lines
                        self.logs.append(json.loads(line.strip()))
        except FileNotFoundError:
            # File does not exist but the first log will create it.
            pass

    async def log(self, data):
        """logs a dictionary as a new line in a jsonl file with a timestamp"""
        if not isinstance(data, dict):
            raise ValueError("Logged data must be a dictionary")
        data["timestamp"] = datetime.now().isoformat()
        self.logs.append(data)
        await self._write_logs_to_file(data)
        print(f"\033[92m{json.dumps(data, indent=4)}\033[0m")

    async def _write_logs_to_file(self, data):
        """appends to the log instead of writing the whole log each time"""
        async with aiofiles.open(self.file_name, "a") as f:
            await f.write(json.dumps(data, indent=None) + "\n")

    def log_sync(self, data: dict):
        """runs the log coroutine non-blocking"""
        asyncio.run_coroutine_threadsafe(self.log(data), self.loop)

    def __repr__(self):
        return f"<AsyncStructuredLogger(file_name={self.file_name})>"
