# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime
from pathlib import Path
import asyncio
import json
import threading

# Third Party
import aiofiles
import aiofiles.os


class AsyncStructuredLogger:
    def __init__(self, file_name="training_log.jsonl"):
        self.file_name = file_name
        self.logs = []
        self.loop = asyncio.new_event_loop()
        self._first_log = True
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
        if aiofiles.path.exists(self.file_name):
            async with aiofiles.open(self.file_name, "r") as f:
                async for line in f:
                    if line.strip():  # Avoid empty lines
                        self.logs.append(json.loads(line.strip()))

    async def log(self, data):
        """logs a dictionary as a new line in a jsonl file with a timestamp"""
        if not isinstance(data, dict):
            raise ValueError("Logged data must be a dictionary")
        data["timestamp"] = datetime.now().isoformat()
        self.logs.append(data)
        await self._write_logs_to_file(data)

    async def _write_logs_to_file(self, data):
        """appends to the log instead of writing the whole log each time"""
        if self._first_log:
            await aiofiles.os.makedirs(Path(self.file_name).parent, exist_ok=True)
            self._first_log = False

        async with aiofiles.open(self.file_name, "a") as f:
            await f.write(json.dumps(data, indent=None) + "\n")

    def log_sync(self, data: dict):
        """runs the log coroutine non-blocking"""
        asyncio.run_coroutine_threadsafe(self.log(data), self.loop)

    def __repr__(self):
        return f"<AsyncStructuredLogger(file_name={self.file_name})>"
