import os
import json
import time
import datetime
import threading
from typing import Dict, List, Optional, Callable
import requests
from anomaly.detectors import AnomalyEvent
from config.loader import AlertingConfig

class AlertManager:
    """
    Manages alert deduplication, cooldown periods, structured JSON logging,
    and webhook dispatching.
    """
    def __init__(self, config: AlertingConfig):
        self.config = config
        self.cooldown_seconds = config.cooldown_seconds
        self.log_file = config.log_file
        self.webhook_url = config.webhook_url
        self.webhook_enabled = config.webhook_enabled
        
        # Track last alert timestamp: key -> timestamp
        self._last_alert_times: Dict[str, float] = {}
        
        # In-memory history of dispatched alerts
        self.dispatched_alerts: List[Dict] = []
        
        # Optional subscriber callbacks (e.g. for WebSocket broadcast)
        self.subscribers: List[Callable[[Dict], None]] = []
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def subscribe(self, callback: Callable[[Dict], None]) -> None:
        self.subscribers.append(callback)

    def dispatch(self, events: List[AnomalyEvent]) -> List[Dict]:
        """
        Processes new anomaly events, applies cooldown deduplication, logs,
        and triggers webhooks for newly approved alerts.
        """
        now = time.time()
        dispatched_this_round = []
        
        for event in events:
            # Construct deduplication key
            target_key = f"{event.camera_id}:{event.anomaly_type}:{event.zone_key or 'global'}"
            if event.track_id is not None:
                target_key += f":{event.track_id}"
                
            last_time = self._last_alert_times.get(target_key, 0.0)
            if (now - last_time) < self.cooldown_seconds:
                # Suppressed by cooldown
                continue
                
            self._last_alert_times[target_key] = now
            
            alert_payload = {
                "id": f"ALT-{int(now * 1000)}-{len(self.dispatched_alerts) + 1}",
                "timestamp": now,
                "datetime": datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc).isoformat(),
                "camera_id": event.camera_id,
                "anomaly_type": event.anomaly_type,
                "severity": event.severity,
                "zone_key": event.zone_key,
                "track_id": event.track_id,
                "message": event.message,
                "metrics": event.metrics,
                "acknowledged": False
            }
            
            # 1. Log to structured JSONL file
            self._log_to_file(alert_payload)
            
            # 2. Add to in-memory history
            self.dispatched_alerts.append(alert_payload)
            dispatched_this_round.append(alert_payload)
            
            # 3. Notify local subscribers
            for cb in self.subscribers:
                try:
                    cb(alert_payload)
                except Exception as e:
                    print(f"[AlertManager] Subscriber callback error: {e}")
                    
            # 4. Trigger Webhook in background thread if enabled
            if self.webhook_enabled and self.webhook_url:
                threading.Thread(target=self._send_webhook, args=(alert_payload,), daemon=True).start()
                
        return dispatched_this_round

    def _log_to_file(self, payload: Dict) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            print(f"[AlertManager] Error logging alert to {self.log_file}: {e}")

    def _send_webhook(self, payload: Dict) -> None:
        try:
            requests.post(self.webhook_url, json=payload, timeout=3.0)
        except Exception as e:
            # Silent fallback so network doesn't interrupt vision pipeline
            pass

    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        return self.dispatched_alerts[-limit:]

    def acknowledge_alert(self, alert_id: str) -> bool:
        for a in self.dispatched_alerts:
            if a["id"] == alert_id:
                a["acknowledged"] = True
                return True
        return False
