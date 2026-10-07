/* ==========================================================================
   RAIL-WATCH AI - Real-Time Dashboard Client Logic
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const fpsDisplay = document.getElementById("fpsDisplay");
  const liveClock = document.getElementById("liveClock");
  const systemStatusText = document.getElementById("systemStatusText");
  const valTotalPassengers = document.getElementById("valTotalPassengers");
  const valPeakDensity = document.getElementById("valPeakDensity");
  const peakTierBadge = document.getElementById("peakTierBadge");
  const valMeanSpeed = document.getElementById("valMeanSpeed");
  const dominantDirText = document.getElementById("dominantDirText");
  const valActiveAlerts = document.getElementById("valActiveAlerts");
  const activeAnomalySummary = document.getElementById("activeAnomalySummary");
  const zonesListContainer = document.getElementById("zonesListContainer");
  const alertsFeedContainer = document.getElementById("alertsFeedContainer");
  const emptyAlertsNotice = document.getElementById("emptyAlertsNotice");
  const totalAlertsBadge = document.getElementById("totalAlertsBadge");
  const videoAnomalyBanner = document.getElementById("videoAnomalyBanner");
  const videoAnomalyText = document.getElementById("videoAnomalyText");
  const radarCanvas = document.getElementById("stationRadarCanvas");
  const radarCtx = radarCanvas ? radarCanvas.getContext("2d") : null;

  // Toggle buttons
  const btnToggleHeatmap = document.getElementById("btnToggleHeatmap");
  const btnToggleROIs = document.getElementById("btnToggleROIs");
  const btnToggleTracks = document.getElementById("btnToggleTracks");
  const btnTogglePrivacy = document.getElementById("btnTogglePrivacy");
  const btnToggleSound = document.getElementById("btnToggleSound");

  // Source buttons
  const btnSourceSim = document.getElementById("btnSourceSim");
  const btnSourceWebcam = document.getElementById("btnSourceWebcam");
  const activeSourceTag = document.getElementById("activeSourceTag");
  const liveVideoFeed = document.getElementById("liveVideoFeed");

  let soundEnabled = false;
  let currentFilter = "all";
  let cachedAlerts = [];
  let socket = null;
  let reconnectInterval = 2000;
  const alertsCard = document.getElementById("alertsCard");
  let lastAlertsSignature = "";
  let knownAlertIds = new Set();

  // Web Audio Context for Chime
  let audioCtx = null;
  function playAlertChime() {
    if (!soundEnabled) return;
    try {
      if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5
      osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.35); // A4
      gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.35);
    } catch (e) {
      console.warn("Audio chime error:", e);
    }
  }

  // Update Live UTC Clock
  function updateClock() {
    const now = new Date();
    liveClock.textContent = now.toTimeString().split(" ")[0] + " UTC";
  }
  setInterval(updateClock, 1000);
  updateClock();

  // Standalone Simulation for Vercel/Static Demo Deployments
  let demoSimulationTimer = null;
  let connectionFailCount = 0;

  function startDemoSimulation() {
    if (demoSimulationTimer) return;
    systemStatusText.textContent = "STANDALONE DEMO";
    systemStatusText.className = "pill-value status-online";
    
    const streamImg = document.getElementById("liveVideoFeed");
    const demoVideo = document.getElementById("demoVideoFeed");
    const fallback = document.getElementById("streamFallback");
    if (streamImg) streamImg.style.display = "none";
    if (fallback) fallback.style.display = "none";
    if (demoVideo) {
      demoVideo.style.display = "block";
      demoVideo.play().catch(() => {});
    }

    demoSimulationTimer = setInterval(() => {
      const mockPax = 12 + Math.floor(Math.sin(Date.now() / 7000) * 5);
      const concoursePax = Math.max(1, Math.floor(mockPax * 0.65));
      const platformPax = Math.max(1, mockPax - concoursePax);
      const concourseDensity = +(concoursePax / 45.0).toFixed(2);
      const platformDensity = +(platformPax / 24.0).toFixed(2);

      const mockVectors = [];
      for (let i = 0; i < mockPax; i++) {
        mockVectors.push({
          x: 220 + (i * 65) % 780 + Math.random() * 30,
          y: 200 + (i * 45) % 380 + Math.random() * 25,
          vx: (Math.random() - 0.5) * 12,
          vy: (Math.random() - 0.5) * 8,
          is_stationary: Math.random() > 0.6
        });
      }

      handleTelemetryUpdate({
        fps: 24.6 + Math.random() * 0.6,
        total_passengers: mockPax,
        motion: {
          mean_speed: 38.5 + Math.random() * 6,
          dominant_direction_deg: 175 + Math.sin(Date.now() / 4000) * 35,
          direction_entropy: 2.1 + Math.random() * 0.25,
          individual_vectors: mockVectors
        },
        zones: {
          waiting_concourse: {
            name: "Waiting Concourse",
            person_count: concoursePax,
            area_m2: 45.0,
            density: concourseDensity,
            tier: concourseDensity > 1.5 ? "Medium" : "Low",
            zone_type: "monitored"
          },
          platform_edge: {
            name: "Platform 1 Edge",
            person_count: platformPax,
            area_m2: 24.0,
            density: platformDensity,
            tier: platformDensity > 3.0 ? "High" : (platformDensity > 1.5 ? "Medium" : "Low"),
            zone_type: "monitored"
          },
          rail_bed_danger_zone: {
            name: "Track Rail Bed (Restricted Zone)",
            person_count: 0,
            area_m2: 12.0,
            density: 0.0,
            tier: "Low",
            zone_type: "restricted"
          }
        },
        active_anomalies: [],
        recent_alerts: cachedAlerts.length > 0 ? cachedAlerts : [
          {
            id: "demo-alert-1",
            severity: "HIGH",
            anomaly_type: "panic_dispersal_monitoring",
            message: "Normal passenger boarding flow monitored at platform edge.",
            timestamp: Math.floor(Date.now() / 1000) - 60,
            camera_id: "CAM-P1-CENTRAL",
            acknowledged: true
          }
        ]
      });
    }, 1500);
  }

  // Connect WebSocket
  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
    
    try {
      socket = new WebSocket(wsUrl);
    } catch (e) {
      connectionFailCount++;
      if (connectionFailCount >= 2 && !demoSimulationTimer) startDemoSimulation();
      return;
    }

    socket.onopen = () => {
      systemStatusText.textContent = "CONNECTED";
      systemStatusText.className = "pill-value status-online";
      reconnectInterval = 2000;
      connectionFailCount = 0;
      if (demoSimulationTimer) {
        clearInterval(demoSimulationTimer);
        demoSimulationTimer = null;
      }
      const streamImg = document.getElementById("liveVideoFeed");
      const demoVideo = document.getElementById("demoVideoFeed");
      if (demoVideo) demoVideo.style.display = "none";
      if (streamImg) streamImg.style.display = "block";
    };

    socket.onmessage = (event) => {
      try {
        const telemetry = JSON.parse(event.data);
        handleTelemetryUpdate(telemetry);
      } catch (err) {
        console.error("Telemetry parse error:", err);
      }
    };

    socket.onclose = () => {
      connectionFailCount++;
      if ((connectionFailCount >= 2 || window.location.hostname.includes("vercel.app")) && !demoSimulationTimer) {
        startDemoSimulation();
      } else if (!demoSimulationTimer) {
        systemStatusText.textContent = "RECONNECTING";
        systemStatusText.className = "pill-value tier-critical";
      }
      setTimeout(connectWebSocket, reconnectInterval);
      reconnectInterval = Math.min(10000, reconnectInterval * 1.5);
    };

    socket.onerror = (err) => {
      connectionFailCount++;
      if ((connectionFailCount >= 2 || window.location.hostname.includes("vercel.app")) && !demoSimulationTimer) {
        startDemoSimulation();
      }
      console.warn("WebSocket error:", err);
    };
  }

  // Process Telemetry Snapshot
  function handleTelemetryUpdate(data) {
    if (!data) return;

    // FPS
    if (data.fps !== undefined) {
      fpsDisplay.textContent = `${data.fps.toFixed(1)} FPS`;
    }

    // Total Passengers
    if (data.total_passengers !== undefined) {
      valTotalPassengers.textContent = data.total_passengers;
    }

    // Motion Metrics
    if (data.motion) {
      valMeanSpeed.textContent = data.motion.mean_speed.toFixed(1);
      dominantDirText.textContent = `Direction: ${data.motion.dominant_direction_deg.toFixed(0)}° (Entropy: ${data.motion.direction_entropy.toFixed(2)})`;
    }

    // Zone Breakdown & Peak Density
    let peakDensity = 0.0;
    let peakTier = "Low";

    if (data.zones) {
      for (const [key, z] of Object.entries(data.zones)) {
        if (z.density > peakDensity && z.zone_type === "monitored") {
          peakDensity = z.density;
          peakTier = z.tier;
        }

        const tierClass = `tier-${z.tier.toLowerCase()}`;
        const maxCapacityRef = 4.5;
        const fillPercent = Math.min(100, Math.round((z.density / maxCapacityRef) * 100));

        let fillColor = "#2ecc71";
        if (z.tier === "Medium") fillColor = "#f1c40f";
        else if (z.tier === "High") fillColor = "#e67e22";
        else if (z.tier === "Critical") fillColor = "#e74c3c";

        let card = document.getElementById(`zoneCard_${key}`);
        if (!card) {
          const div = document.createElement("div");
          div.className = "zone-item-card";
          div.id = `zoneCard_${key}`;
          div.innerHTML = `
            <div class="zone-header-row">
              <span class="zone-title">${z.name}</span>
              <span class="tier-pill ${tierClass}" id="tierPill_${key}">${z.tier}</span>
            </div>
            <div class="zone-meta-row">
              <div class="zone-numbers">
                <span class="zone-density-val" id="densityVal_${key}">${z.density.toFixed(2)}</span>
                <span>p/m²</span>
              </div>
              <span id="paxInfo_${key}">${z.person_count} pax / ${z.area_m2}m²</span>
            </div>
            <div class="zone-gauge-bar-bg">
              <div class="zone-gauge-fill" id="gaugeFill_${key}" style="width: ${fillPercent}%; background-color: ${fillColor};"></div>
            </div>
          `;
          const placeholder = zonesListContainer.querySelector(".loading-placeholder");
          if (placeholder) placeholder.remove();
          zonesListContainer.appendChild(div);
        } else {
          const pill = document.getElementById(`tierPill_${key}`);
          if (pill) {
            pill.textContent = z.tier;
            pill.className = `tier-pill ${tierClass}`;
          }
          const dVal = document.getElementById(`densityVal_${key}`);
          if (dVal) dVal.textContent = z.density.toFixed(2);
          const pInfo = document.getElementById(`paxInfo_${key}`);
          if (pInfo) pInfo.textContent = `${z.person_count} pax / ${z.area_m2}m²`;
          const gauge = document.getElementById(`gaugeFill_${key}`);
          if (gauge) {
            gauge.style.width = `${fillPercent}%`;
            gauge.style.backgroundColor = fillColor;
          }
        }
      }
    }

    valPeakDensity.textContent = peakDensity.toFixed(2);
    peakTierBadge.textContent = peakTier.toUpperCase();
    peakTierBadge.className = `tier-pill tier-${peakTier.toLowerCase()}`;

    // Active Anomalies & Banner
    const hasActiveAnomalies = !!(data.active_anomalies && data.active_anomalies.length > 0);
    
    // Flicker / pulse the Security & Safety Incident Feed ONLY when an anomaly is active
    if (alertsCard) {
      alertsCard.classList.toggle("anomaly-flicker", hasActiveAnomalies);
    }

    if (hasActiveAnomalies) {
      valActiveAlerts.textContent = data.active_anomalies.length;
      activeAnomalySummary.textContent = `${data.active_anomalies.length} active incidents!`;
      activeAnomalySummary.style.color = "var(--accent-rose)";

      const topAnom = data.active_anomalies[0];
      videoAnomalyText.textContent = `[${topAnom.severity}] ${topAnom.message}`;
      videoAnomalyBanner.style.display = "flex";

      if (topAnom.severity === "CRITICAL") {
        playAlertChime();
      }
    } else {
      valActiveAlerts.textContent = "0";
      activeAnomalySummary.textContent = "System Normal";
      activeAnomalySummary.style.color = "var(--text-secondary)";
      videoAnomalyBanner.style.display = "none";
    }

    // Recent Alerts Feed
    if (data.recent_alerts) {
      cachedAlerts = data.recent_alerts;
      renderAlertsList();
    }

    // Render Radar Canvas
    if (radarCtx && data.motion && data.motion.individual_vectors) {
      renderRadar(data.motion.individual_vectors, data.zones);
    }
  }

  // Render Alerts List with Filters - Zero constant flicker, updates only on state change
  function renderAlertsList(force = false) {
    let filtered = cachedAlerts;
    if (currentFilter !== "all") {
      filtered = cachedAlerts.filter(a => a.severity === currentFilter);
    }

    totalAlertsBadge.textContent = cachedAlerts.length;

    // Check if alerts state actually changed
    const signature = `${currentFilter}::` + filtered.map(a => `${a.id}:${a.acknowledged}`).join("|");
    if (!force && signature === lastAlertsSignature) {
      return; // Feed content is identical - do NOT modify DOM, preventing flicker!
    }
    lastAlertsSignature = signature;

    if (!filtered || filtered.length === 0) {
      emptyAlertsNotice.style.display = "flex";
      alertsFeedContainer.querySelectorAll(".alert-item").forEach(el => el.remove());
      return;
    }

    emptyAlertsNotice.style.display = "none";

    let html = "";
    // Display in reverse chronological order (newest first)
    const reversed = [...filtered].reverse();
    for (const alert of reversed) {
      const sevClass = alert.severity === "CRITICAL" ? "crit" : (alert.severity === "HIGH" ? "high" : "med");
      const isNew = !knownAlertIds.has(alert.id);
      const newClass = isNew ? " new-arrival" : "";
      const timeFormatted = new Date(alert.timestamp * 1000).toTimeString().split(" ")[0];
      const ackClass = alert.acknowledged ? "ack-btn acked" : "ack-btn";
      const ackText = alert.acknowledged ? "✓ Acknowledged" : "Acknowledge";

      html += `
        <div class="alert-item ${sevClass}${newClass}" id="alertCard_${alert.id}">
          <div class="alert-item-header">
            <span class="alert-sev-tag ${sevClass}">${alert.severity} • ${alert.anomaly_type.replace('_', ' ')}</span>
            <span class="alert-time">${timeFormatted}</span>
          </div>
          <p class="alert-message">${alert.message}</p>
          <div class="alert-footer">
            <span class="alert-metrics-snippet">CAM: ${alert.camera_id}</span>
            <button class="${ackClass}" onclick="acknowledgeAlert('${alert.id}')">${ackText}</button>
          </div>
        </div>
      `;
    }

    // Mark current alerts as known so only future arrivals get new-arrival animation
    filtered.forEach(a => knownAlertIds.add(a.id));

    alertsFeedContainer.innerHTML = html;
  }

  // Acknowledge Alert Handler
  window.acknowledgeAlert = async function(alertId) {
    try {
      const res = await fetch(`/api/alerts/${alertId}/acknowledge`, { method: "POST" });
      if (res.ok) {
        const found = cachedAlerts.find(a => a.id === alertId);
        if (found) found.acknowledged = true;
        renderAlertsList(true);
      }
    } catch (err) {
      console.error("Acknowledge error:", err);
    }
  };

  // Filter Buttons
  document.querySelectorAll(".filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentFilter = btn.getAttribute("data-filter");
      renderAlertsList(true);
    });
  });

  // Source Switching Logic
  async function switchSource(sourceValue, isWebcam) {
    try {
      activeSourceTag.textContent = "Switching camera feed...";
      const res = await fetch("/api/source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: sourceValue })
      });
      const data = await res.json();
      
      if (btnSourceSim && btnSourceWebcam) {
        btnSourceSim.classList.toggle("active", !isWebcam);
        btnSourceWebcam.classList.toggle("active", isWebcam);
      }
      
      activeSourceTag.textContent = isWebcam ? "Current Source: Live Webcam (0)" : "Current Source: Station Entry Live Feed";
      
      // Refresh video feed stream
      if (liveVideoFeed) {
        liveVideoFeed.src = "/api/stream?t=" + new Date().getTime();
      }
    } catch (err) {
      console.error("Source switch error:", err);
      activeSourceTag.textContent = "Error switching source";
    }
  }

  if (btnSourceSim) {
    btnSourceSim.addEventListener("click", () => switchSource("data/station_entry_live.mp4", false));
  }
  if (btnSourceWebcam) {
    btnSourceWebcam.addEventListener("click", () => switchSource("0", true));
  }

  // Stream Toggle Buttons
  async function bindToggle(btn, setting) {
    if (!btn) return;
    btn.addEventListener("click", async () => {
      try {
        const res = await fetch(`/api/toggle/${setting}`, { method: "POST" });
        const json = await res.json();
        const isActive = json[setting];
        btn.classList.toggle("active", isActive);
      } catch (err) {
        console.error(`Toggle error for ${setting}:`, err);
      }
    });
  }

  bindToggle(btnToggleHeatmap, "heatmap");
  bindToggle(btnToggleROIs, "polygons");
  bindToggle(btnToggleTracks, "tracks");
  bindToggle(btnTogglePrivacy, "privacy");

  if (btnToggleSound) {
    btnToggleSound.addEventListener("click", () => {
      soundEnabled = !soundEnabled;
      btnToggleSound.classList.toggle("active", soundEnabled);
      if (soundEnabled) playAlertChime();
    });
  }

  // Draw 2D Radar Canvas
  function renderRadar(vectors, zones) {
    const w = radarCanvas.width;
    const h = radarCanvas.height;

    radarCtx.clearRect(0, 0, w, h);

    // Coordinate scale factors from 1280x720 CCTV frame
    const sx = w / 1280;
    const sy = h / 720;

    // Draw Grid Lines (Subtle azure blue grid on light background)
    radarCtx.strokeStyle = "rgba(2, 132, 199, 0.12)";
    radarCtx.lineWidth = 1;
    for (let x = 0; x < w; x += 40) {
      radarCtx.beginPath();
      radarCtx.moveTo(x, 0);
      radarCtx.lineTo(x, h);
      radarCtx.stroke();
    }
    for (let y = 0; y < h; y += 40) {
      radarCtx.beginPath();
      radarCtx.moveTo(0, y);
      radarCtx.lineTo(w, y);
      radarCtx.stroke();
    }

    // Draw Platform Safety Line
    radarCtx.strokeStyle = "rgba(217, 119, 6, 0.85)";
    radarCtx.lineWidth = 3;
    radarCtx.beginPath();
    radarCtx.moveTo(1080 * sx, 0);
    radarCtx.lineTo(1080 * sx, h);
    radarCtx.stroke();

    // Draw Tracks Restricted Zone (Alert tint)
    radarCtx.fillStyle = "rgba(239, 68, 68, 0.12)";
    radarCtx.fillRect(1100 * sx, 0, w - 1100 * sx, h);

    // Draw Monitored Zone Outlines
    if (zones) {
      for (const [k, z] of Object.entries(zones)) {
        if (k === "waiting_concourse") {
          radarCtx.strokeStyle = "rgba(5, 150, 105, 0.6)";
          radarCtx.lineWidth = 1.5;
          radarCtx.strokeRect(80 * sx, 160 * sy, 620 * sx, 500 * sy);
        } else if (k === "platform_edge") {
          radarCtx.strokeStyle = "rgba(2, 132, 199, 0.6)";
          radarCtx.lineWidth = 1.5;
          radarCtx.strokeRect(720 * sx, 160 * sy, 360 * sx, 500 * sy);
        }
      }
    }

    // Draw Passenger Blips
    for (const v of vectors) {
      const px = v.x * sx;
      const py = v.y * sy;

      // Draw Motion vector tail
      radarCtx.strokeStyle = "rgba(2, 132, 199, 0.7)";
      radarCtx.lineWidth = 1.5;
      radarCtx.beginPath();
      radarCtx.moveTo(px, py);
      radarCtx.lineTo(px + (v.vx * 0.3), py + (v.vy * 0.3));
      radarCtx.stroke();

      // Blip circle
      radarCtx.fillStyle = v.is_stationary ? "#d97706" : "#0284c7";
      radarCtx.shadowColor = "rgba(2, 132, 199, 0.5)";
      radarCtx.shadowBlur = 5;
      radarCtx.beginPath();
      radarCtx.arc(px, py, 3.5, 0, Math.PI * 2);
      radarCtx.fill();
      radarCtx.shadowBlur = 0;
    }
  }

  // Handle Stream Disconnects
  let streamFailCount = 0;
  window.handleStreamError = function() {
    streamFailCount++;
    const streamImg = document.getElementById("liveVideoFeed");
    const demoVideo = document.getElementById("demoVideoFeed");
    const fallback = document.getElementById("streamFallback");

    if (streamFailCount >= 2 || window.location.hostname.includes("vercel.app")) {
      if (streamImg) streamImg.style.display = "none";
      if (fallback) fallback.style.display = "none";
      if (demoVideo) {
        demoVideo.style.display = "block";
        demoVideo.play().catch(() => {});
      }
      if (!demoSimulationTimer) startDemoSimulation();
      return;
    }

    if (streamImg) streamImg.style.display = "none";
    if (fallback) fallback.style.display = "flex";
    setTimeout(() => {
      if (streamImg) {
        streamImg.src = "/api/stream?t=" + new Date().getTime();
        streamImg.style.display = "block";
      }
      if (fallback) fallback.style.display = "none";
    }, 3000);
  };

  // Start WebSocket
  connectWebSocket();
});
