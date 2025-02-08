from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from queue import Queue
import os
import sqlite3
import threading
import time

# FastAPI app
app = FastAPI(title="Event Logger Service", version="1.0.0", description="Event Logger with Producer-Consumer Model")

# Database setup
DB_NAME = os.getenv("LOGS_DB_NAME", "../../db.sqlite3")

class Event(BaseModel):
    TS: int  # UNIX Timestamp (in nanoseconds)
    PID: int  # Process ID
    TYPE: str  # Event type ('O', 'C', 'D', 'E')
    FLAG: Optional[str] = None
    PATTERN: Optional[str] = None
    OPEN: int = 0
    CREATE: int = 0
    DELETE: int = 0
    ENCRYPT: int = 0
    FILENAME: Optional[str] = None

# Queue for events
EVENT_WRITE_QUEUE = Queue()

# DB Writer (Consumer)
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 100))
BATCH_INTERVAL = int(os.getenv('BATCH_INTERVAL', 1))  # seconds

def db_writer_batch():
    """Consumer thread that writes events in batches to SQLite."""
    batch = []
    last_flush_time = time.time()

    while True:
        try:
            event = EVENT_WRITE_QUEUE.get(timeout=1)
            if event is None:  # Graceful shutdown
                if batch:
                    flush_batch(batch)
                break

            batch.append(event)

            # Flush batch if size exceeds BATCH_SIZE or time interval elapses
            if len(batch) >= BATCH_SIZE or time.time() - last_flush_time >= BATCH_INTERVAL:
                flush_batch(batch)
                batch = []  # Reset batch
                last_flush_time = time.time()

            EVENT_WRITE_QUEUE.task_done()

        except Exception as e:
            print(f"Error: {type(e)}")
        except:
            continue  # Timeout when queue is empty

def flush_batch(batch):
    """Write a batch of events to SQLite."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT INTO events (TS, PID, TYPE, FLAG, PATTERN, \'OPEN\', \'CREATE\', \'DELETE\', \'ENCRYPT\', FILENAME)
            VALUES (strftime(\'%s\', \'now\'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [(e['PID'], e['TYPE'], e.get('FLAG'), e.get('PATTERN'),
               e.get('OPEN', 0), e.get('CREATE', 0), e.get('DELETE', 0),
               e.get('ENCRYPT', 0), e.get('FILENAME')) for e in batch])
        conn.commit()
        conn.close()
        print(f"Flushed {len(batch)} events to DB.")
    except Exception as e:
        print(f"Failed to flush batch: {e}")
    

# Start DB writer in a background thread
consumer_thread = threading.Thread(target=db_writer_batch, daemon=True)
consumer_thread.start()

# API Route (Producer)
@app.post("/events", summary="Send event to the logger", description="Send an event to be logged into the system.")
async def send_event(event: Event):
    """
    API route to accept events and add them to the in-memory queue.
    """
    try:
        # Convert event to dict and add to queue
        event_data = event.dict()
        EVENT_WRITE_QUEUE.put(event_data)
        return {"message": "Event received and queued successfully.", "event": event_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error queuing event: {str(e)}")

# Graceful shutdown
@app.on_event("shutdown")
def shutdown_event():
    """Graceful shutdown for the consumer."""
    EVENT_WRITE_QUEUE.put(None)
    consumer_thread.join()

# Run with: uvicorn filename:app --reload

