/**
 * CodeAlpha Object Detection & Tracking Web Dashboard Controller
 * Pure vanilla JavaScript: handles telemetry polling, stream controls, file uploads, and UI updates.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const connectionDot = document.getElementById("connection-dot");
    const connectionText = document.getElementById("connection-text");
    const sessionBadge = document.getElementById("session-badge");
    const statusBanner = document.getElementById("status-banner");

    // Controls
    const srcWebcam = document.getElementById("src-webcam");
    const srcFile = document.getElementById("src-file");
    const uploadBox = document.getElementById("upload-box");
    const videoFileInput = document.getElementById("video-file-input");
    const dropZone = document.getElementById("drop-zone");
    const dropText = document.getElementById("drop-text");
    const uploadedFileBadge = document.getElementById("uploaded-file-badge");
    const uploadProgress = document.getElementById("upload-progress");
    const confSlider = document.getElementById("conf-slider");
    const confVal = document.getElementById("conf-val");
    const toggleLine = document.getElementById("toggle-line");
    const toggleRoi = document.getElementById("toggle-roi");
    const toggleHeatmap = document.getElementById("toggle-heatmap");
    const btnStart = document.getElementById("btn-start");
    const btnStop = document.getElementById("btn-stop");
    const streamSourceBadge = document.getElementById("stream-source-badge");

    // KPI Metrics
    const kpiFps = document.getElementById("kpi-fps");
    const kpiActive = document.getElementById("kpi-active");
    const kpiTotal = document.getElementById("kpi-total");
    const kpiFrames = document.getElementById("kpi-frames");

    // Analytics Panels
    const sceneSummaryText = document.getElementById("scene-summary-text");
    const classTagsContainer = document.getElementById("class-tags-container");
    const spatialCrossingsTotal = document.getElementById("spatial-crossings-total");
    const spatialCrossingsDir = document.getElementById("spatial-crossings-dir");
    const roiActiveOccupants = document.getElementById("roi-active-occupants");
    const roiPeakOccupancy = document.getElementById("roi-peak-occupancy");
    const behaviorTracksCount = document.getElementById("behavior-tracks-count");
    const behaviorAvgLength = document.getElementById("behavior-avg-length");
    const behaviorAvgDisp = document.getElementById("behavior-avg-disp");
    const behaviorAvgTort = document.getElementById("behavior-avg-tort");
    const eventTableBody = document.getElementById("event-table-body");
    const btnExportHeatmap = document.getElementById("btn-export-heatmap");

    // State Variables
    let currentUploadedFilePath = null;
    let isStreaming = false;
    let pollInterval = null;

    // -------------------------------------------------------------
    // EVENT LISTENERS & FORM HANDLING
    // -------------------------------------------------------------

    confVal.textContent = parseFloat(confSlider.value).toFixed(2);

    confSlider.addEventListener("input", (e) => {
        confVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    srcWebcam.addEventListener("change", () => {
        uploadBox.classList.add("hidden");
        streamSourceBadge.textContent = "Source: Webcam (0)";
    });

    srcFile.addEventListener("change", () => {
        uploadBox.classList.remove("hidden");
        if (currentUploadedFilePath) {
            streamSourceBadge.textContent = `Source: File (${currentUploadedFilePath.split(/[\/\\]/).pop()})`;
        } else {
            streamSourceBadge.textContent = "Source: Upload Video File";
        }
    });

    // File Drop Zone Handling
    dropZone.addEventListener("click", () => videoFileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--accent-blue)";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "var(--border-color)";
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--border-color)";
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    videoFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    function showBanner(message, type = "info") {
        statusBanner.textContent = message;
        statusBanner.className = `notification-banner ${type}`;
        statusBanner.classList.remove("hidden");
    }

    function hideBanner() {
        statusBanner.classList.add("hidden");
    }

    // -------------------------------------------------------------
    // FILE UPLOAD HANDLER
    // -------------------------------------------------------------
    async function handleFileUpload(file) {
        hideBanner();

        const allowedVideoExts = [".mp4", ".avi", ".mov", ".mkv", ".webm"];
        const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
        if (!allowedVideoExts.includes(ext)) {
            showBanner(`Unsupported file format '${ext}'. Please upload a video file (.mp4, .avi, .mov, .mkv, .webm).`, "error");
            return;
        }

        uploadProgress.classList.remove("hidden");
        dropText.textContent = `Selected: ${file.name}`;

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            const data = await res.json();

            uploadProgress.classList.add("hidden");
            if (res.ok && data.success) {
                currentUploadedFilePath = data.filepath;
                uploadedFileBadge.textContent = `Uploaded: ${data.original_name}`;
                uploadedFileBadge.classList.remove("hidden");
                streamSourceBadge.textContent = `Source: File (${data.original_name})`;
                showBanner("Video uploaded successfully. Ready to stream.", "success");
            } else {
                showBanner(data.message || "Upload failed.", "error");
                currentUploadedFilePath = null;
            }
        } catch (err) {
            uploadProgress.classList.add("hidden");
            showBanner(`Upload network error: ${err.message}`, "error");
            currentUploadedFilePath = null;
        }
    }

    // -------------------------------------------------------------
    // STREAM LIFECYCLE (START / STOP)
    // -------------------------------------------------------------
    btnStart.addEventListener("click", async () => {
        hideBanner();
        btnStart.disabled = true;

        let source = "0";
        if (srcFile.checked) {
            if (!currentUploadedFilePath) {
                showBanner("Please upload or select a video file first.", "error");
                btnStart.disabled = false;
                return;
            }
            source = currentUploadedFilePath;
        }

        const payload = {
            source: source,
            conf_threshold: parseFloat(confSlider.value),
            enable_heatmap: toggleHeatmap.checked,
            virtual_line: toggleLine.checked ? [0, 240, 640, 240] : null,
            roi_region: toggleRoi.checked ? [100, 100, 540, 380] : null
        };

        try {
            const res = await fetch("/api/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (res.ok && data.success) {
                isStreaming = true;
                btnStart.disabled = true;
                btnStop.disabled = false;
                showBanner("Detection stream started successfully.", "success");
                sessionBadge.textContent = "RUNNING";
                sessionBadge.className = "session-badge running";
            } else {
                showBanner(data.message || "Failed to start stream.", "error");
                btnStart.disabled = false;
            }
        } catch (err) {
            showBanner(`Network error: ${err.message}`, "error");
            btnStart.disabled = false;
        }
    });

    btnStop.addEventListener("click", async () => {
        btnStop.disabled = true;
        showBanner("Stopping stream...", "info");

        try {
            const res = await fetch("/api/stop", { method: "POST" });
            const data = await res.json();

            if (res.ok && data.success) {
                isStreaming = false;
                btnStart.disabled = false;
                btnStop.disabled = true;
                showBanner("Stream stopped cleanly.", "success");
                sessionBadge.textContent = "STOPPED";
                sessionBadge.className = "session-badge";
            } else {
                showBanner(data.message || "Failed to stop stream.", "error");
                btnStop.disabled = false;
            }
        } catch (err) {
            showBanner(`Network error: ${err.message}`, "error");
            btnStop.disabled = false;
        }
    });

    // -------------------------------------------------------------
    // TELEMETRY POLLING
    // -------------------------------------------------------------
    async function pollTelemetry() {
        try {
            const res = await fetch("/api/telemetry");
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();

            // Connection indicator
            connectionDot.classList.add("online");
            connectionText.textContent = "Connected";

            // Session state updates
            if (data.status === "running") {
                sessionBadge.textContent = "RUNNING";
                sessionBadge.className = "session-badge running";
                btnStart.disabled = true;
                btnStop.disabled = false;
                isStreaming = true;
            } else if (data.status === "stopping") {
                sessionBadge.textContent = "STOPPING";
                sessionBadge.className = "session-badge stopping";
                btnStart.disabled = true;
                btnStop.disabled = true;
            } else if (data.status === "completed") {
                sessionBadge.textContent = "COMPLETED";
                sessionBadge.className = "session-badge completed";
                btnStart.disabled = false;
                btnStop.disabled = true;
                isStreaming = false;
            } else {
                sessionBadge.textContent = (data.status || "IDLE").toUpperCase();
                sessionBadge.className = "session-badge";
                btnStart.disabled = false;
                btnStop.disabled = true;
                isStreaming = false;
            }

            // Top KPI metrics
            kpiFps.innerHTML = `${(data.fps || 0.0).toFixed(1)} <span class="kpi-unit">FPS</span>`;
            kpiActive.textContent = data.active_tracks_count || 0;
            kpiTotal.textContent = data.total_unique_tracks || 0;
            kpiFrames.textContent = data.frames_processed || 0;

            // Scene Summary
            if (data.scene_summary) {
                sceneSummaryText.textContent = data.scene_summary;
            }

            // Visible Class Badges
            if (data.visible_classes && Object.keys(data.visible_classes).length > 0) {
                classTagsContainer.innerHTML = Object.entries(data.visible_classes)
                    .map(([cls, cnt]) => `<span class="class-tag">${cls}: ${cnt}</span>`)
                    .join("");
            } else {
                classTagsContainer.innerHTML = '<span class="class-tag" style="color:var(--text-secondary);">No objects currently in frame</span>';
            }

            // Spatial Analytics Panel
            if (data.spatial) {
                spatialCrossingsTotal.textContent = data.spatial.total_line_crossings || 0;
                if (data.spatial.crossings_by_direction) {
                    const dirs = Object.entries(data.spatial.crossings_by_direction)
                        .map(([d, c]) => `${d}: ${c}`).join(", ");
                    spatialCrossingsDir.textContent = dirs || "None";
                }
                roiActiveOccupants.textContent = (data.spatial.current_roi_occupants && data.spatial.current_roi_occupants.length) || 0;
                roiPeakOccupancy.textContent = (data.spatial.peak_roi_occupancy && data.spatial.peak_roi_occupancy.count) || 0;
            }

            // Behavior Analytics Panel
            if (data.behavior) {
                behaviorTracksCount.textContent = data.behavior.total_tracks_analyzed || 0;
                behaviorAvgLength.textContent = `${(data.behavior.avg_path_length_px || 0.0).toFixed(1)} px`;
                behaviorAvgDisp.textContent = `${(data.behavior.avg_net_displacement_px || 0.0).toFixed(1)} px`;
                behaviorAvgTort.textContent = (data.behavior.avg_tortuosity || 1.0).toFixed(2);
            }

            // Event Activity Log Table
            if (data.recent_events && data.recent_events.length > 0) {
                eventTableBody.innerHTML = data.recent_events.slice(-6).reverse().map(ev => {
                    let typeClass = "";
                    const et = (ev.event_type || "").toUpperCase();
                    if (et === "ENTERED") typeClass = "tag-entered";
                    else if (et === "MOVED") typeClass = "tag-moved";
                    else if (et === "EXITED") typeClass = "tag-exited";
                    else if (et.includes("CROSS")) typeClass = "tag-crossing";

                    return `
                        <tr>
                            <td>${(ev.timestamp_sec || 0).toFixed(1)}s</td>
                            <td>${ev.frame_idx || 0}</td>
                            <td>#${ev.track_id || 0} ${ev.class_name || ""}</td>
                            <td class="${typeClass}">${ev.event_type || ""}</td>
                            <td>${ev.description || ""}</td>
                        </tr>
                    `;
                }).join("");
            }

        } catch (err) {
            connectionDot.classList.remove("online");
            connectionText.textContent = "Disconnected";
        }
    }

    // Start polling loop: 600ms when streaming, 1200ms when idle
    pollInterval = setInterval(pollTelemetry, 800);
    pollTelemetry();
});
