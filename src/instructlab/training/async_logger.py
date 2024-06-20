# File: async_logger.py

import json
import asyncio
from datetime import datetime
import aiofiles
import threading

class AsyncStructuredLogger:
    def __init__(self, file_name='training_log.json'):
        self.file_name = file_name
        self.logs = []
        self.loop = asyncio.new_event_loop()
        t = threading.Thread(target=self._run_event_loop, args=(self.loop,), daemon=True)
        t.start()
        asyncio.run_coroutine_threadsafe(self._initialize_log_file(), self.loop)

    def _run_event_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _initialize_log_file(self):
        try:
            async with aiofiles.open(self.file_name, 'r') as f:
                self.logs = json.loads(await f.read())
        except FileNotFoundError:
            async with aiofiles.open(self.file_name, 'w') as f:
                await f.write(json.dumps(self.logs, indent=4))

    async def log(self, data):
        if not isinstance(data, dict):
            raise ValueError("Logged data must be a dictionary")
        data['timestamp'] = datetime.now().isoformat()
        self.logs.append(data)
        await self._write_logs_to_file()

    async def _write_logs_to_file(self):
        async with aiofiles.open(self.file_name, 'w') as f:
            await f.write(json.dumps(self.logs, indent=4))

    def log_sync(self, data: dict):
        asyncio.run_coroutine_threadsafe(self.log(data), self.loop)

    def __repr__(self):
        return f"<AsyncStructuredLogger(file_name={self.file_name})>"
