
from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

import serial
import serial.tools.list_ports


class SerialBridge:
    def __init__(self, port: str, baud_rate: int) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        self._callbacks: List[Callable[[str], None]] = []
        self._lock = threading.Lock()

    def list_ports(self) -> List[str]:
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self, port: str, baud_rate: Optional[int] = None) -> bool:
        self.port = port
        if baud_rate is not None:
            self.baud_rate = baud_rate
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=0.25)
            time.sleep(2.0)
            self.connected = True
            threading.Thread(target=self._read_loop, daemon=True).start()
            self.send("STATUS?")
            return True
        except Exception:
            self.connected = False
            return False

    def disconnect(self) -> None:
        self.connected = False
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    def send(self, cmd: str) -> None:
        if not self.connected or not self.ser:
            return
        with self._lock:
            try:
                self.ser.write((cmd.strip() + "\n").encode("utf-8", errors="ignore"))
            except Exception:
                self.connected = False

    def add_callback(self, cb: Callable[[str], None]) -> None:
        self._callbacks.append(cb)

    def _read_loop(self) -> None:
        while self.connected and self.ser:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode(errors="ignore").strip()
                    if line:
                        for cb in list(self._callbacks):
                            cb(line)
            except Exception:
                self.connected = False
                break
            time.sleep(0.02)
